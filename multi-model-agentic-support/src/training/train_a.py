"""Continue the existing dataset and DistilBERT task; select only on validation."""
import gc
import math
import numpy as np
from src.config import ROOT, BASE_AB, LABELS
from src.training.common import read, save, split_intents, seed
from src.evaluation.metrics import classification, threshold_sweep

def main(epochs=8):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
    seed()
    rows = split_intents(read('intents.json'))
    tokenizer = AutoTokenizer.from_pretrained(BASE_AB)
    datasets = [Dataset.from_list(r).map(lambda x: tokenizer(x['text'], truncation=True, max_length=128), batched=True) for r in rows]
    def new_model():
        return AutoModelForSequenceClassification.from_pretrained(BASE_AB, num_labels=len(LABELS),
                    id2label=dict(enumerate(LABELS)), label2id={v: i for i, v in enumerate(LABELS)})
    model = new_model()
    baseline = Trainer(model=model, args=TrainingArguments(output_dir=str(ROOT/'models/baseline_a'), report_to='none'),
                       data_collator=DataCollatorWithPadding(tokenizer))
    bval = baseline.predict(datasets[1])
    baseline_validation = classification(bval.label_ids, bval.predictions.argmax(-1), list(range(8)))
    del baseline
    def metrics(p):
        result = classification(p.label_ids, p.predictions.argmax(-1), list(range(8)))
        return {k: v for k, v in result.items() if isinstance(v, float)}
    args = TrainingArguments(output_dir=str(ROOT/'models/a_checkpoints'), learning_rate=5e-5, num_train_epochs=epochs,
        per_device_train_batch_size=8, per_device_eval_batch_size=8, weight_decay=0.01, warmup_steps=max(1, math.ceil(len(datasets[0])/8*epochs*.1)),
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model='f1_macro', greater_is_better=True, report_to='none', seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=datasets[0], eval_dataset=datasets[1],
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    trainer.train()
    save('model_a_training_log.json', trainer.state.log_history)
    val = trainer.predict(datasets[1])
    vp = torch.tensor(val.predictions).softmax(-1).numpy()
    sweep = threshold_sweep(val.label_ids.tolist(), vp.argmax(-1).tolist(), vp.max(-1).tolist())
    save('threshold.json', sweep)
    # Final test is opened only after model/threshold selection has finished.
    test = trainer.predict(datasets[2]); tp = torch.tensor(test.predictions).softmax(-1).numpy()
    result = classification(test.label_ids, tp.argmax(-1), list(range(8)))
    errors = [{'text': r['text'], 'true': LABELS[r['label']], 'predicted': LABELS[int(p)]}
              for r, p in zip(rows[2], tp.argmax(-1)) if r['label'] != p]
    path = ROOT/'models/intent_classifier'
    trainer.save_model(str(path)); tokenizer.save_pretrained(path)
    sample = tokenizer('PostgreSQL connections are exhausted.', return_tensors='pt')
    model.eval()
    with torch.inference_mode(): original = model(**{k:v.to(model.device) for k,v in sample.items()}).logits.cpu()
    reload = AutoModelForSequenceClassification.from_pretrained(path).eval()
    with torch.inference_mode(): reloaded = reload(**sample).logits
    reload_ok = bool(torch.allclose(original, reloaded, atol=1e-4))
    del reload, model, trainer; gc.collect()
    seed(); baseline = new_model().eval()
    bt = Trainer(model=baseline, args=TrainingArguments(output_dir=str(ROOT/'models/baseline_a'), report_to='none'), data_collator=DataCollatorWithPadding(tokenizer)).predict(datasets[2])
    result.update({'baseline_validation': baseline_validation, 'baseline_test': classification(bt.label_ids, bt.predictions.argmax(-1), list(range(8))),
                   'errors': errors, 'reload_ok': reload_ok, 'artifact': str(path), 'threshold': sweep,
                   'gate': result['f1_macro'] >= 0.8 and min(result['per_class_recall'].values()) >= 0.6 and reload_ok})
    save('model_a.json', result)
    return result

if __name__ == '__main__':
    print(main())
