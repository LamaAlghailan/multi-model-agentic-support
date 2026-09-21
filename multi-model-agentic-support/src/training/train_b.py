"""Original document split and span mechanics, with validation-based model selection."""
import gc
import math
from src.config import ROOT, BASE_AB
from src.training.common import read, save, split_qa, seed
from src.evaluation.metrics import qa_scores
from src.models.qa_model import best_span

def features(tokenizer, rows):
    batch = tokenizer([r['question'].strip() for r in rows], [r['context'] for r in rows],
                      truncation='only_second', max_length=384, stride=96, return_overflowing_tokens=True,
                      return_offsets_mapping=True, padding='max_length')
    samples = batch.pop('overflow_to_sample_mapping')
    offsets = batch.pop('offset_mapping')
    starts, ends, metadata = [], [], []
    for i, sample in enumerate(samples):
        row = rows[sample]
        start = row['answer_start']; end = start + len(row['answer_text'])
        if row['context'][start:end] != row['answer_text']:
            raise ValueError('Invalid literal answer span')
        valid = [j for j, s in enumerate(batch.sequence_ids(i)) if s == 1]
        cls = batch['input_ids'][i].index(tokenizer.cls_token_id)
        if not valid or offsets[i][valid[0]][0] > start or offsets[i][valid[-1]][1] < end:
            starts.append(cls); ends.append(cls)
        else:
            starts.append(next(j for j in valid if offsets[i][j][1] > start))
            ends.append(next(j for j in reversed(valid) if offsets[i][j][0] < end))
        metadata.append({'sample': sample, 'offsets': [o if s == 1 else None for o, s in zip(offsets[i], batch.sequence_ids(i))]})
    batch['start_positions'] = starts; batch['end_positions'] = ends
    return dict(batch), metadata

def evaluate(trainer, dataset, metadata, rows):
    output = trainer.predict(dataset)
    start_logits, end_logits = output.predictions[:2]
    candidates = [[] for _ in rows]
    for i, meta in enumerate(metadata):
        candidates[meta['sample']].append(best_span(rows[meta['sample']]['context'], meta['offsets'], start_logits[i], end_logits[i]))
    details = []
    for row, spans in zip(rows, candidates):
        prediction = max(spans, key=lambda x:x['score'])['text']
        details.append({'id': row['id'], 'prediction': prediction, 'gold': row['answer_text'],
                        'windows': len(spans), **qa_scores(prediction, row['answer_text'])})
    return {**{k:sum(r[k] for r in details)/len(details) for k in ['exact_match', 'token_f1']}, 'details': details}

def main(epochs=12):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments, DefaultDataCollator
    seed()
    rows = split_qa(read('qa.json'))
    tokenizer = AutoTokenizer.from_pretrained(BASE_AB)
    prepared = [features(tokenizer, r) for r in rows]
    datasets = [Dataset.from_dict(p[0]) for p in prepared]
    model = AutoModelForQuestionAnswering.from_pretrained(BASE_AB)
    baseline = Trainer(model=model, args=TrainingArguments(output_dir=str(ROOT/'models/baseline_b'), report_to='none'), data_collator=DefaultDataCollator())
    baseline_val = evaluate(baseline, datasets[1], prepared[1][1], rows[1])
    del baseline
    args = TrainingArguments(output_dir=str(ROOT/'models/b_checkpoints'), learning_rate=3e-5, num_train_epochs=epochs,
            per_device_train_batch_size=8, per_device_eval_batch_size=8, weight_decay=0.01, warmup_steps=max(1, math.ceil(len(datasets[0])/8*epochs*.1)),
            eval_strategy='epoch', save_strategy='epoch', save_total_limit=1, load_best_model_at_end=True,
            metric_for_best_model='eval_loss', greater_is_better=False, report_to='none', seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=datasets[0], eval_dataset=datasets[1], data_collator=DefaultDataCollator())
    trainer.train()
    save('model_b_training_log.json', trainer.state.log_history)
    val = evaluate(trainer, datasets[1], prepared[1][1], rows[1])
    result = evaluate(trainer, datasets[2], prepared[2][1], rows[2])
    path = ROOT/'models/qa_model'; trainer.save_model(str(path)); tokenizer.save_pretrained(path)
    sample = tokenizer('Which port?', 'The application listens on port 8000.', return_tensors='pt')
    model.eval()
    with torch.inference_mode(): original = model(**{k:v.to(model.device) for k,v in sample.items()}).start_logits.cpu()
    reloaded = AutoModelForQuestionAnswering.from_pretrained(path).eval()
    with torch.inference_mode(): reload_ok = bool(torch.allclose(original, reloaded(**sample).start_logits, atol=1e-4))
    del reloaded, trainer, model; gc.collect()
    seed(); baseline_model = AutoModelForQuestionAnswering.from_pretrained(BASE_AB)
    baseline = Trainer(model=baseline_model, args=TrainingArguments(output_dir=str(ROOT/'models/baseline_b'), report_to='none'), data_collator=DefaultDataCollator())
    result.update({'baseline_validation': baseline_val, 'validation': val, 'baseline_test': evaluate(baseline, datasets[2], prepared[2][1], rows[2]),
        'reload_ok': reload_ok, 'artifact': str(path), 'gate': result['exact_match'] >= .65 and result['token_f1'] >= .8 and reload_ok})
    save('model_b.json', result)
    return result

if __name__ == '__main__':
    print(main())
