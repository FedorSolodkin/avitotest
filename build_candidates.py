"""
Кандидатогенерация: для каждого запроса собираем пул из ~2000 объявлений-кандидатов и
сохраняем скор по каждому источнику похожести — это вход для ранкера (train_and_predict.py).

Источники похожести:
  * TF-IDF по словам и по символьным n-граммам отдельно для заголовка и для "тела"
    объявления (заголовок + параметры + начало описания) — n-граммы ловят опечатки
    и разные словоформы, которых не поймать по словам;
  * TF-IDF по символьным n-граммам между запросом и текстами запросов, по которым
    объявление уже выбирали в train ("doc expansion" — расширение документа историей
    выбора: если объявление раньше выбирали по запросу "массаж спины", это сильный
    сигнал, даже если само объявление слова "спина" не содержит);
  * косинус между плотными эмбеддингами (compute_embeddings.py);
  * априорная вероятность локации объявления при данной локации поиска, оценённая по train.

Кандидатный пул = объединение топ-300 по каждому источнику (с бонусом за локацию и без)
плюс топ-1000 по их взвешенной сумме. Веса взвешенной суммы подобраны перебором
на локальной валидации.

Запуск: python build_candidates.py val   — пул для локальной валидации
        python build_candidates.py bench — пул для запросов бенчмарка
"""
import gc
import os
import re
import sys
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

DATA_DIR = 'data'
CACHE_DIR = 'cache'

MODE = sys.argv[1] if len(sys.argv) > 1 else 'val'
assert MODE in ('val', 'bench'), 'режим должен быть val или bench'

# итоговые веса линейной комбинации скоров, подобранные перебором на локальной валидации
BASE_WEIGHTS = dict(body_word=1.0, body_char=1.33, hist_char=0.7, embed=3.54)
FINAL_LOC_WEIGHT = 0.1125   # вес локации в итоговой линейной комбинации
POOL_LOC_BONUS = 0.15       # более щедрый бонус за локацию при отборе кандидатов в пул
POOL_PER_FIELD = 300        # топ-N кандидатов от каждого источника
POOL_TOTAL = 1000           # + топ-N по взвешенной сумме


def norm(s):
    s = (s or '').lower().replace('ё', 'е')
    return re.sub(r'[^0-9a-zа-я]+', ' ', s)


def fit_sim(docs, queries, cache_name, analyzer, ngram_range):
    """TF-IDF поверх HashingVectorizer: словарь не хранится в памяти, а векторизация
    идёт кусками — корпус большой (189 тыс. объявлений). Матрица объявлений кэшируется
    на диск, чтобы не пересчитывать её при повторных запусках; для текстовых полей,
    которые не зависят от train (заголовок, тело), кэш общий для val и bench."""
    docs, queries = list(docs), list(queries)
    item_only = cache_name.startswith('title') or cache_name.startswith('body')
    cache_path = os.path.join(CACHE_DIR, 'tfidf_%s%s.npz' % (cache_name, '' if item_only else '_' + MODE))

    hv = HashingVectorizer(n_features=2 ** 21, alternate_sign=False, norm=None, dtype=np.float32,
                            analyzer=analyzer, ngram_range=ngram_range, token_pattern=r'\w+')
    if os.path.exists(cache_path):
        item_counts = sp.load_npz(cache_path)
    else:
        item_counts = sp.vstack([hv.transform(docs[i:i + 20000]) for i in range(0, len(docs), 20000)]).tocsr()
        sp.save_npz(cache_path, item_counts, compressed=False)

    tf = TfidfTransformer(sublinear_tf=True).fit(item_counts)
    item_mat = tf.transform(item_counts).astype(np.float32)
    query_mat = tf.transform(hv.transform(queries)).astype(np.float32)
    return item_mat.T.tocsr(), query_mat.tocsr()   # транспонируем один раз, а не в каждом батче


def build_location_prior(train_pairs, item_loc_values):
    """P(локация объявления | локация поиска) по train, со сглаживанием. Если такую
    локацию поиска в train не видели — небольшой бонус за точное совпадение локаций,
    без штрафа для остальных (не хотим наказывать за незнакомую локацию)."""
    loc_counts = defaultdict(Counter)
    for search_loc, item_loc in zip(train_pairs['search_location_id'].values,
                                     train_pairs['item_location_id'].values):
        loc_counts[search_loc][item_loc] += 1

    uniq_loc, idx = np.unique(item_loc_values, return_inverse=True)

    def logprior(search_loc):
        counts = loc_counts.get(search_loc)
        prob = np.full(len(uniq_loc), 1e-4, dtype=np.float32)
        if counts:
            total = sum(counts.values())
            for loc, n in counts.items():
                j = np.searchsorted(uniq_loc, loc)
                if j < len(uniq_loc) and uniq_loc[j] == loc:
                    prob[j] += n / total
        else:
            j = np.searchsorted(uniq_loc, search_loc)
            if j < len(uniq_loc) and uniq_loc[j] == search_loc:
                prob[j] += 0.8
            else:
                prob[:] = 1.0   # ничего не знаем про эту локацию — без штрафа
        return np.log(prob)[idx]

    return logprior


def main():
    items = pd.read_pickle(os.path.join(CACHE_DIR, 'items.pkl'))
    item_ids = items['item_id'].values
    item_loc = items['item_location_id'].values

    if MODE == 'val':
        queries = pd.read_pickle(os.path.join(CACHE_DIR, 'val_queries.pkl')).reset_index(drop=True)
        train_pairs = pd.read_pickle(os.path.join(CACHE_DIR, 'train_pairs_excl_val.pkl'))
    else:
        queries = pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_queries.parquet')).reset_index(drop=True)
        train_pairs = pd.read_pickle(os.path.join(CACHE_DIR, 'train_pairs.pkl'))
    queries['search_infm_params_text'] = queries['search_infm_params_text'].fillna('')

    title = items['item_title_raw'].map(norm)
    body = (title + ' ' + items['params_clean'].map(norm) + ' ' + items['desc_short'].str.slice(0, 800).map(norm))
    q_text = queries['search_query'].map(norm)

    hist = train_pairs.groupby('item_id')['q'].apply(lambda x: ' '.join(x.map(norm)))
    q_hist = items['item_id'].map(hist).fillna('')

    mats = {
        'title_word': fit_sim(title, q_text, 'title_word', 'word', (1, 2)),
        'title_char': fit_sim(title, q_text, 'title_char', 'char_wb', (3, 5)),
        'body_word': fit_sim(body, q_text, 'body_word', 'word', (1, 1)),
        'body_char': fit_sim(body, q_text, 'body_char', 'char_wb', (4, 4)),
        'hist_char': fit_sim(q_hist, q_text, 'hist_char', 'char_wb', (3, 5)),
    }
    del title, body, q_hist
    gc.collect()

    embed_items = np.load(os.path.join(CACHE_DIR, 'emb_items.npy')).astype(np.float32)
    embed_queries = np.load(os.path.join(CACHE_DIR, 'emb_q_%s.npy' % MODE)).astype(np.float32)
    fields = list(mats) + ['embed']

    logprior = build_location_prior(train_pairs, item_loc)
    search_loc = queries['search_location_id'].values

    rows = []
    batch = 256
    for start in range(0, len(queries), batch):
        end = min(start + batch, len(queries))
        scores = {f: (mats[f][1][start:end] @ mats[f][0]).toarray() for f in mats}
        scores['embed'] = embed_queries[start:end] @ embed_items.T

        for r in range(end - start):
            lp = logprior(search_loc[start + r])
            pool = set()
            for f in fields:
                s = scores[f][r]
                pool |= set(np.argpartition(-s, POOL_PER_FIELD)[:POOL_PER_FIELD])
                pool |= set(np.argpartition(-(s + POOL_LOC_BONUS * lp), POOL_PER_FIELD)[:POOL_PER_FIELD])
            base = sum(scores[f][r] * w for f, w in BASE_WEIGHTS.items())
            pool |= set(np.argpartition(-(base + FINAL_LOC_WEIGHT * lp), POOL_TOTAL)[:POOL_TOTAL])
            pool = np.array(sorted(pool))
            feat = np.stack([scores[f][r][pool] for f in fields] + [lp[pool]], axis=1).astype(np.float32)
            rows.append((pool, feat))
        del scores
        gc.collect()

    out_path = os.path.join(CACHE_DIR, 'pool_%s.pkl' % MODE)
    with open(out_path, 'wb') as fh:
        pickle.dump((fields, rows), fh)

    pool_sizes = [len(p) for p, _ in rows]
    print('средний размер пула кандидатов:', np.mean(pool_sizes))
    if MODE == 'val':
        rel = queries['rel'].values
        recall = np.mean([len(set(item_ids[p]) & r) / len(r) for (p, _), r in zip(rows, rel)])
        print('полнота пула (доля релевантных объявлений, попавших в кандидаты): %.4f' % recall)


if __name__ == '__main__':
    main()
