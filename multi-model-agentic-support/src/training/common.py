import json
import random
from src.config import ROOT

def read(name):
    return json.loads((ROOT / 'data' / name).read_text(encoding='utf-8'))

def save(name, value):
    path = ROOT / 'reports' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def split_intents(rows):
    from sklearn.model_selection import train_test_split
    train, rest = train_test_split(rows, test_size=48, random_state=42, stratify=[r['label'] for r in rows])
    val, test = train_test_split(rest, test_size=24, random_state=42, stratify=[r['label'] for r in rest])
    return train, val, test

def split_qa(rows):
    val = {'docker', 'tickets'}
    test = {'long_context_a', 'long_context_b'}
    return ([r for r in rows if r['doc_id'] not in val | test],
            [r for r in rows if r['doc_id'] in val], [r for r in rows if r['doc_id'] in test])

def split_sft(rows):
    # Preserve original category hold-outs. New contrastive pairs are training-only.
    import pandas as pd
    train, val, test = [], [], []
    df = pd.DataFrame(rows)
    for _, group in df.groupby('category'):
        group = group.sample(frac=1, random_state=42).to_dict('records')
        test.append(group[0]); val.append(group[1]); train.extend(group[2:])
    return train, val, test

def seed():
    import torch
    from transformers import set_seed
    set_seed(42)
    # Avoid oversubscribing small CPU training workloads.
    torch.set_num_threads(min(4, torch.get_num_threads()))
