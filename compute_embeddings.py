"""
Плотные эмбеддинги текста объявлений и запросов моделью intfloat/multilingual-e5-small
(open-source, работает локально на CPU, без обращений к внешним API во время инференса;
веса модели скачиваются с Hugging Face один раз при первом запуске и дальше берутся
из локального кэша).

TF-IDF/хеш-признаки (build_candidates.py) ловят точные совпадения слов и опечатки,
а эмбеддинги — смысловую близость (синонимы, другой порядок слов, другую словоформу),
поэтому это два взаимодополняющих источника кандидатов.

Запуск: python compute_embeddings.py
"""
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = 'data'
CACHE_DIR = 'cache'

MODEL_NAME = 'intfloat/multilingual-e5-small'
MAX_SEQ_LEN = 48   # заголовок + короткий хвост параметров, длиннее не нужно


def encode(model, texts, batch_size=128):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True)


def main():
    torch.set_num_threads(os.cpu_count() or 8)
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    model.max_seq_length = MAX_SEQ_LEN

    items = pd.read_pickle(os.path.join(CACHE_DIR, 'items.pkl'))
    docs = ('passage: ' + items['item_title_raw'] + '. ' + items['params_clean'].str.slice(0, 120)).tolist()
    # сортируем по длине текста, чтобы батчи были плотнее (меньше паддинга — быстрее на CPU)
    order = np.argsort([len(d) for d in docs])
    emb = encode(model, [docs[i] for i in order]).astype(np.float16)
    item_emb = np.empty_like(emb)
    item_emb[order] = emb
    np.save(os.path.join(CACHE_DIR, 'emb_items.npy'), item_emb)

    val = pd.read_pickle(os.path.join(CACHE_DIR, 'val_queries.pkl'))
    q_emb = encode(model, ('query: ' + val['search_query'].str.lower()).tolist())
    np.save(os.path.join(CACHE_DIR, 'emb_q_val.npy'), q_emb.astype(np.float16))

    bench = pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_queries.parquet'))
    q_emb = encode(model, ('query: ' + bench['search_query'].str.lower()).tolist())
    np.save(os.path.join(CACHE_DIR, 'emb_q_bench.npy'), q_emb.astype(np.float16))

    print('готово')


if __name__ == '__main__':
    main()
