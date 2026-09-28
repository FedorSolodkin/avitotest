"""
Финальный этап: обучаем LightGBM (lambdarank) ранжировать кандидатов из build_candidates.py
и сохраняем топ-50 объявлений на каждый запрос бенчмарка в answer.csv.

Помимо скоров-кандидатов (TF-IDF/эмбеддинги/локация из build_candidates.py) модель получает:
  * относительный ранг и относительную величину каждого скора внутри запроса
    (сырые TF-IDF-скоры разных запросов между собой не сравнимы, а ранг — сравним);
  * долю объявлений с такой же подкатегорией среди топ-50 по базовому скору
    (псевдо-обратная связь: если запросу отвечает конкретная подкатегория, это сигнал);
  * совпадение локации объявления с локацией поиска;
  * долю пересечения фильтров поиска (search_infm_params_text) с параметрами объявления;
  * прямое пересечение слов запроса с заголовком/полным текстом объявления —
    дешёвый, но сильный сигнал именно для "хвостовых" запросов, где TF-IDF-скор
    почти одинаков у многих объявлений;
  * статические признаки объявления: рейтинг, число отзывов, цена, категория и т.п.

Качество проверяем 2-fold кросс-валидацией по запросам локальной валидации
(build_candidates.py val) — это единственная метрика, которой можно доверять локально,
поскольку у бенчмарка разметки нет.

Запуск: python train_and_predict.py
Библиотеки: lightgbm, pandas, numpy.
"""
import os
import pickle
import re

import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_DIR = 'data'
CACHE_DIR = 'cache'
KEEP = 1000          # после скоринга оставляем top-KEEP кандидатов на запрос перед подачей в LightGBM
N_FOLDS = 2
TOP_K = 50

LGB_PARAMS = dict(
    objective='lambdarank', metric='map', eval_at=[TOP_K],
    learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambdarank_truncation_level=60, verbose=-1, num_threads=os.cpu_count() or 8,
)
N_ROUNDS = 400

# веса той же линейной комбинации скоров, что и в build_candidates.py — нужны здесь,
# чтобы отранжировать кандидатов внутри запроса перед обрезкой пула до KEEP
BASE_WEIGHTS = dict(body_word=1.0, body_char=1.33, hist_char=0.7, embed=3.54)
LOC_WEIGHT = 0.1125


def norm_tokens(s):
    return set(re.sub(r'[^0-9a-zа-я]+', ' ', (s or '').lower().replace('ё', 'е')).split())


def load_items():
    items = pd.read_pickle(os.path.join(CACHE_DIR, 'items.pkl'))
    static = pd.DataFrame({
        'rating': items['item_rating'].fillna(-1).values,
        'reviews': np.log1p(items['item_rating_reviews_count'].fillna(0).values),
        'price': np.log1p(items['item_price'].fillna(0).clip(lower=0).values),
        'phone_hidden': items['item_is_phone_hidden'].astype(float).values,
        'msg_forbidden': items['item_is_message_forbidden'].astype(float).values,
        'microcat': items['item_microcat_id'].values,
        'is_cat114': (items['item_category_id'] == 114).astype(float).values,
        'title_len': items['item_title_raw'].str.len().values,
    }).astype(np.float32)
    title_tok = [norm_tokens(s) for s in items['item_title_raw'].values]
    params_tok = [norm_tokens(s) for s in items['params_clean'].values]
    desc_tok = [norm_tokens(s) for s in items['desc_short'].values]
    title_norm = [re.sub(r'[^0-9a-zа-я]+', ' ', (s or '').lower().replace('ё', 'е'))
                  for s in items['item_title_raw'].values]
    return items, static, title_tok, params_tok, desc_tok, title_norm


ITEMS, STATIC_FEAT, TITLE_TOK, PARAMS_TOK, DESC_TOK, TITLE_NORM = load_items()
ITEM_IDS = ITEMS['item_id'].values
ITEM_LOC = ITEMS['item_location_id'].values


def build_features(mode):
    with open(os.path.join(CACHE_DIR, 'pool_%s.pkl' % mode), 'rb') as fh:
        fields, rows = pickle.load(fh)
    if mode == 'val':
        queries = pd.read_pickle(os.path.join(CACHE_DIR, 'val_queries.pkl'))
    else:
        queries = pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_queries.parquet'))
    queries = queries.reset_index(drop=True)

    col_names = fields + ['loc_lp']
    feats, labels, groups, cand_idx = [], [], [], []
    for qi, (pool, raw) in enumerate(rows):
        f = pd.DataFrame(raw, columns=col_names)
        for c in fields:
            m = f[c].max()
            f[c + '_rel'] = f[c] / m if m > 0 else 0.0
            f[c + '_rank'] = (-f[c]).rank(method='first').astype(np.float32)

        base = sum(f[c] * w for c, w in BASE_WEIGHTS.items()) + LOC_WEIGHT * f['loc_lp']
        f['base'] = base
        # оставляем топ-KEEP по базовому скору: экономим память, полнота пула почти не теряется
        keep = np.sort(np.argsort(-base.values)[:KEEP])
        f, pool, base = f.iloc[keep].reset_index(drop=True), pool[keep], base.iloc[keep].reset_index(drop=True)
        f['base_rank'] = (-base).rank(method='first')

        # доля объявлений с такой же подкатегорией среди топ-50 по базовому скору
        microcat = STATIC_FEAT['microcat'].values[pool]
        top = np.argsort(-base.values)[:TOP_K]
        share = pd.Series(microcat[top]).value_counts(normalize=True)
        f['microcat_share'] = pd.Series(microcat).map(share).fillna(0).values

        search_loc = queries['search_location_id'].values[qi]
        f['loc_eq'] = (ITEM_LOC[pool] == search_loc).astype(np.float32)

        filters = norm_tokens(queries['search_infm_params_text'].values[qi])
        f['has_filter'] = float(len(filters) > 0)
        f['filter_match'] = [len(filters & PARAMS_TOK[j]) / len(filters) if filters else 0.0 for j in pool]

        q_raw = str(queries['search_query'].values[qi])
        q_tok = norm_tokens(q_raw)
        n_q = max(1, len(q_tok))
        f['q_len'] = len(q_raw.split())
        f['title_cover'] = [len(q_tok & TITLE_TOK[j]) / n_q for j in pool]
        f['all_cover'] = [len(q_tok & (TITLE_TOK[j] | PARAMS_TOK[j] | DESC_TOK[j])) / n_q for j in pool]
        q_norm = re.sub(r'[^0-9a-zа-я]+', ' ', q_raw.lower().replace('ё', 'е')).strip()
        f['title_substr'] = [float(q_norm in TITLE_NORM[j]) for j in pool] if q_norm else 0.0

        f = pd.concat([f, STATIC_FEAT.iloc[pool].reset_index(drop=True)], axis=1)
        feats.append(f.astype(np.float32))
        if mode == 'val':
            labels.append(np.isin(ITEM_IDS[pool], list(queries['rel'].values[qi])).astype(int))
        groups.append(len(pool))
        cand_idx.append(pool)

    X = pd.concat(feats, ignore_index=True)
    y = np.concatenate(labels) if labels else None
    return X, y, np.array(groups), cand_idx, queries


def predict_top_k(model, X, groups, cand_idx, k=TOP_K):
    scores = model.predict(X)
    result, offset = [], 0
    for g, pool in zip(groups, cand_idx):
        s = scores[offset:offset + g]
        offset += g
        result.append(ITEM_IDS[pool[np.argsort(-s)[:k]]])
    return result


def recall_at_k(preds, relevant_sets):
    return np.mean([len(set(p) & r) / len(r) for p, r in zip(preds, relevant_sets)])


def cross_validate(X, y, groups, cand_idx, relevant_sets, n_folds=N_FOLDS):
    """K-fold по запросам (не по строкам!): в каждом фолде модель не видит запросы,
    на которых потом считаем recall — иначе оценка будет завышенной."""
    offsets = np.concatenate([[0], np.cumsum(groups)])
    n_queries = len(groups)
    fold_of = np.arange(n_queries) % n_folds
    preds = [None] * n_queries
    for k in range(n_folds):
        train_q, test_q = np.where(fold_of != k)[0], np.where(fold_of == k)[0]
        train_rows = np.concatenate([np.arange(offsets[q], offsets[q + 1]) for q in train_q])
        test_rows = np.concatenate([np.arange(offsets[q], offsets[q + 1]) for q in test_q])
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X.iloc[train_rows], y[train_rows], group=groups[train_q]), N_ROUNDS)
        fold_preds = predict_top_k(model, X.iloc[test_rows], groups[test_q], [cand_idx[q] for q in test_q])
        for q, p in zip(test_q, fold_preds):
            preds[q] = p
    return recall_at_k(preds, relevant_sets)


def main():
    Xv, yv, gv, cand_v, val_queries = build_features('val')
    print('признаки модели:', list(Xv.columns))

    baseline_rank = -Xv['base_rank'].values
    baseline_preds = []
    offset = 0
    for g, pool in zip(gv, cand_v):
        s = baseline_rank[offset:offset + g]
        offset += g
        baseline_preds.append(ITEM_IDS[pool[np.argsort(-s)[:TOP_K]]])
    print('recall@50 без ранкера (просто по базовому скору): %.4f' % recall_at_k(baseline_preds, val_queries['rel']))

    cv_recall = cross_validate(Xv, yv, gv, cand_v, val_queries['rel'])
    print('recall@50 после ранкера LightGBM, %d-fold CV: %.4f' % (N_FOLDS, cv_recall))

    # финальная модель — обучаем на всей валидации, предсказываем на бенчмарке
    final_model = lgb.train(LGB_PARAMS, lgb.Dataset(Xv, yv, group=gv), N_ROUNDS)
    importance = pd.Series(final_model.feature_importance('gain'), index=Xv.columns).sort_values(ascending=False)
    print('важность признаков (топ-10):')
    print(importance.head(10))
    del Xv

    Xb, _, gb, cand_b, bench_queries = build_features('bench')
    bench_preds = predict_top_k(final_model, Xb, gb, cand_b)

    answer = pd.DataFrame({
        'query_id': bench_queries['query_id'].astype(str),
        'answer': [' '.join(p) for p in bench_preds],
    })
    answer.to_csv('answer.csv', index=False)
    print('сохранено answer.csv:', len(answer), 'строк')


if __name__ == '__main__':
    main()
