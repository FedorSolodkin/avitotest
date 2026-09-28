"""Проверка answer.csv на соответствие формату задания перед отправкой."""
import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else 'answer.csv'
answer = pd.read_csv(path, dtype=str, keep_default_na=False)
queries = pd.read_parquet('data/benchmark_queries.parquet')
item_ids = set(pd.read_parquet('data/benchmark_items.parquet', columns=['item_id'])['item_id'])

assert list(answer.columns) == ['query_id', 'answer'], 'должны быть ровно колонки query_id, answer'
assert answer['query_id'].is_unique, 'есть повторяющиеся query_id'
assert set(answer['query_id']) == set(queries['query_id']), 'набор query_id не совпадает с benchmark_queries'

lists = answer['answer'].str.split(' ')
assert lists.map(len).max() <= 50, 'больше 50 item_id в одной строке'
assert (lists.map(len) == lists.map(lambda x: len(set(x)))).all(), 'есть повторы item_id внутри строки'
assert all(item_id in item_ids for row in lists for item_id in row), 'есть item_id не из benchmark_items'

print(path, 'OK:', len(answer), 'строк, в среднем', lists.map(len).mean(), 'id на запрос')
