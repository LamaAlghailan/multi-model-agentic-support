"""Explicit assistant token labels, independent of TRL chat-template mask support."""

def assistant_features(tokenizer, messages, max_length=512):
    if len(messages) != 3 or [m['role'] for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError('Training expects one system/user/assistant conversation')
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    pids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    ids = tokenizer(full, add_special_tokens=False)['input_ids']
    if ids[:len(pids)] != pids:
        raise ValueError('Chat-template token boundary is not prefix-stable')
    if len(ids) > max_length:
        raise ValueError('Conversation exceeds training context; do not silently truncate the target')
    labels = [-100] * len(pids) + ids[len(pids):]
    if not any(v != -100 for v in labels):
        raise ValueError('No assistant tokens to train on')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}

class AssistantCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        import torch
        size = max(len(ex['input_ids']) for ex in examples)
        return {key: torch.tensor([ex[key] + [pad] * (size - len(ex[key])) for ex in examples])
                for key, pad in [('input_ids', self.tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}
