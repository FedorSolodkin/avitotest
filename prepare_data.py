"""
Готовит промежуточные данные для пайплайна:
  1. очищает тексты объявлений (заголовок, параметры, начало описания) -> cache/items.pkl
  2. собирает пары "запрос-объявление" из train.parquet в компактном виде -> cache/train_pairs.pkl
  3. выкраивает из train локальную валидацию, имитирующую распределение бенчмарка
     (в бенчмарке 63% запросов вообще не встречаются в train, у остальных медианная
     частота встречаемости — 3 раза) -> cache/val_queries.pkl, cache/train_pairs_excl_val.pkl

Запуск: python prepare_data.py
"""
import os
import re

import numpy as np
import pandas as pd

DATA_DIR = 'data'
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

N_VAL_UNSEEN = 1900   # запросы, которых в train не остаётся вовсе после исключения валидации
N_VAL_RARE = 1100      # запросы редкие (<=30 раз), но не единственный раз


def clean_params(s):
    """Из строки параметров объявления выбрасываем служебный шум (расписание, типовые
    ценовые поля) и оставляем содержательные слова: вид/тип услуги, названия услуг,
    специальность, марки и т.п."""
    s = re.sub(r'(Время для связи|График работы|Время работы)[^А-ЯA-Z]*', ' ', s)
    s = re.sub(r'Дни (пн|вт|ср|чт|пт|сб|вс)', ' ', s)
    s = re.sub(r'Рабочие дни \S+', ' ', s)
    s = re.sub(r'(Начальная цена|Тип стоимости за (услугу|час)|Стоимость \d+|'
               r'Продолжительность( \d+ ?(ч|мин\.))*)', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def prepare_items():
    it = pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_items.parquet'))
    for c in ['item_title_raw', 'item_infm_params_text', 'item_description_raw']:
        it[c] = it[c].fillna('')
    it['item_latitude'] = it['item_latitude'].astype(float)
    it['item_longitude'] = it['item_longitude'].astype(float)
    it['item_price'] = it['item_price'].astype(float)
    it['params_clean'] = it['item_infm_params_text'].map(clean_params)
    it['desc_short'] = it['item_description_raw'].str.slice(0, 1500)
    it = it.drop(columns=['item_description_raw', 'item_infm_params_text'])
    it.to_pickle(os.path.join(CACHE_DIR, 'items.pkl'))
    return it


def load_train_pairs():
    cols = ['search_query', 'search_location_id', 'search_infm_params_text',
            'search_category', 'item_id', 'item_location_id']
    tr = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'), columns=cols)
    tr['q'] = tr['search_query'].str.lower().str.strip()
    tr.to_pickle(os.path.join(CACHE_DIR, 'train_pairs.pkl'))
    return tr


def build_validation(tr, item_ids):
    """Локальная валидация нужна, чтобы честно замерять Recall@50 до отправки на Stepik.
    Если брать под валидацию случайные запросы train, они окажутся сильно "легче"
    бенчмарка — в среднем встречаются в train десятки раз, тогда как в бенчмарке
    63% запросов не встречаются вовсе. Поэтому специально подбираем такое же
    соотношение "незнакомых" и "редких" запросов."""
    in_corpus = tr['item_id'].isin(item_ids)
    keys = tr[in_corpus].groupby(['q', 'search_location_id']).size().reset_index()[['q', 'search_location_id']]
    keys = keys.sample(frac=1, random_state=0).drop_duplicates('q')

    q_freq = tr['q'].value_counts()
    q_ngroups = tr.groupby('q')['search_location_id'].nunique()
    keys['freq'] = keys['q'].map(q_freq)
    keys['ngroups'] = keys['q'].map(q_ngroups)

    unseen = keys[keys['ngroups'] == 1].head(N_VAL_UNSEEN)                       # после отложения текста в train не останется
    rare = keys[(keys['ngroups'] > 1) & (keys['freq'] <= 30)].head(N_VAL_RARE)   # редкие, но встречаются и в других локациях
    val_keys = set(map(tuple, pd.concat([unseen, rare])[['q', 'search_location_id']].values))

    tr = tr.copy()
    tr['key'] = list(zip(tr['q'], tr['search_location_id']))
    is_val = tr['key'].isin(val_keys)

    val = (tr[is_val & in_corpus]
           .groupby(['q', 'search_location_id'])
           .agg(search_query=('search_query', 'first'),
                search_infm_params_text=('search_infm_params_text', 'first'),
                search_category=('search_category', 'first'),
                rel=('item_id', lambda x: set(x)))
           .reset_index())
    val['query_id'] = ['val_%d' % i for i in range(len(val))]

    val.to_pickle(os.path.join(CACHE_DIR, 'val_queries.pkl'))
    tr[~is_val].drop(columns=['key']).to_pickle(os.path.join(CACHE_DIR, 'train_pairs_excl_val.pkl'))
    return val, int(is_val.sum())


if __name__ == '__main__':
    items = prepare_items()
    pairs = load_train_pairs()
    val, n_val_rows = build_validation(pairs, set(items['item_id']))
    print('объявлений в корпусе:', len(items))
    print('пар запрос-объявление в train:', len(pairs))
    print('запросов в локальной валидации:', len(val),
          '| среднее число релевантных объявлений на запрос:', val['rel'].map(len).mean())
    print('строк train, отложенных под валидацию:', n_val_rows)
