"""Nonquantized baseline -> cleanup -> fresh QLoRA/LoRA with assistant-only labels."""
import gc
import importlib.util
import math
from src.config import ROOT, BASE_C
from src.training.common import read, save, split_sft, seed
from src.training.assistant_loss import assistant_features, AssistantCollator
from src.evaluation.metrics import rouge_l
from src.evaluation.golden import evaluate as evaluate_golden

class Generator:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    def generate(self, messages, max_new_tokens=180):
        import torch
        self.model.eval()
        if messages[-1]['role'] == 'assistant': messages = messages[:-1]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

def generations(model, tokenizer, rows):
    generator = Generator(model, tokenizer)
    return [{'category': r['category'], 'reference': r['messages'][-1]['content'],
             'prediction': generator.generate(r['messages'])} for r in rows]

def main():
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    seed()
    original = read('sft_original.json')
    train, val, test = split_sft(original)
    train += read('sft_contrastive.json')
    golden = read('golden_set.json')
    golden_prompts = {c['messages'][-1]['content'] for c in golden}
    # The original fixture has exact Golden Set overlaps. Keep the prompts fixed, exclude training overlaps.
    excluded = [r for r in train if r['messages'][1]['content'] in golden_prompts]
    train = [r for r in train if r['messages'][1]['content'] not in golden_prompts]
    from src.models.tokenizer import load_support_tokenizer
    tokenizer = load_support_tokenizer(BASE_C)
    tokenizer.pad_token = tokenizer.eos_token
    datasets = [Dataset.from_list([assistant_features(tokenizer, r['messages']) for r in rows]) for rows in [train,val,test]]
    collator = AssistantCollator(tokenizer)
    baseline_model = AutoModelForCausalLM.from_pretrained(BASE_C)
    baseline_trainer = Trainer(model=baseline_model, args=TrainingArguments(output_dir=str(ROOT/'models/baseline_c'),
                              per_device_eval_batch_size=2, report_to='none'), data_collator=collator)
    baseline_eval = baseline_trainer.evaluate(datasets[1])
    baseline_loss = float(baseline_eval['eval_loss'])
    baseline_outputs = generations(baseline_model, tokenizer, test)
    baseline_rouge = sum(rouge_l(r['prediction'],r['reference']) for r in baseline_outputs)/len(test)
    save('model_c_baseline.json', {'loss': baseline_loss, 'perplexity': math.exp(baseline_loss), 'rouge_l': baseline_rouge,
                                  'generations': baseline_outputs, 'base': BASE_C, 'assistant_only': True})
    del baseline_trainer, baseline_model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    use_qlora = torch.cuda.is_available() and importlib.util.find_spec('bitsandbytes') is not None
    if use_qlora:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_C, quantization_config=config, device_map='auto')
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(BASE_C)
    model = get_peft_model(base_model, LoraConfig(task_type='CAUSAL_LM', r=16, lora_alpha=32, lora_dropout=.05, target_modules=['q_proj','v_proj'], bias='none'))
    args = TrainingArguments(output_dir=str(ROOT/'models/c_checkpoints'), num_train_epochs=5, per_device_train_batch_size=2,
             per_device_eval_batch_size=2, gradient_accumulation_steps=8, learning_rate=2e-4, lr_scheduler_type='cosine',
             warmup_steps=max(1, math.ceil(len(train)/16*5*.1)), weight_decay=.01, max_grad_norm=1., eval_strategy='epoch', save_strategy='epoch',
             load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, save_total_limit=1,
             bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), report_to='none', seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=datasets[0], eval_dataset=datasets[1], data_collator=collator)
    trainer.train()
    save('model_c_training_log.json', trainer.state.log_history)
    loss = float(trainer.evaluate(datasets[1])['eval_loss'])
    test_loss = float(trainer.evaluate(datasets[2])['eval_loss'])
    outputs = generations(trainer.model, tokenizer, test)
    rouge = sum(rouge_l(r['prediction'],r['reference']) for r in outputs)/len(test)
    golden_result = evaluate_golden(Generator(trainer.model, tokenizer), golden)
    path = ROOT/'models/support_adapter'; trainer.save_model(str(path)); tokenizer.save_pretrained(path)
    before = Generator(trainer.model,tokenizer).generate(test[0]['messages'])
    del trainer, model, base_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE_C), path).eval()
    after = Generator(reloaded, tokenizer).generate(test[0]['messages'])
    result = {'base': BASE_C, 'artifact': str(path), 'assistant_only': True, 'training_mode': 'qlora' if use_qlora else 'lora',
              'train_count': len(train), 'validation_count':len(val), 'test_count':len(test), 'excluded_exact_golden_overlaps':len(excluded),
              'baseline_loss':baseline_loss, 'loss':loss, 'baseline_perplexity':math.exp(baseline_loss), 'perplexity':math.exp(loss),
              'test_loss':test_loss, 'baseline_rouge_l':baseline_rouge, 'rouge_l':rouge, 'generations':outputs, 'golden':golden_result,
              'reload_ok':before == after, 'gate': loss < baseline_loss and golden_result['gate'] and before == after,
              'limitations':['Golden Set is a fixed development regression set; original data includes near-duplicates.',
                            'ROUGE-L uses whitespace tokens without stemming; not numerically comparable to historical rouge-score.']}
    save('model_c.json',result)
    return result

if __name__ == '__main__':
    print(main())
