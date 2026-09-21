"""Restore legacy Hub chat templates when the tokenizer loader omits them."""
import json
from pathlib import Path

def load_support_tokenizer(source):
    from transformers import AutoTokenizer
    # Download the required metadata explicitly before tokenizer construction. A partial
    # optional-file fetch must not silently produce a tokenizer without control tokens.
    if Path(source).is_dir():
        path = Path(source)/'tokenizer_config.json'
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(source, 'tokenizer_config.json'))
    config = json.loads(path.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    if not tokenizer.chat_template:
        template = config.get('chat_template')
        if not isinstance(template, str) or not template:
            raise ValueError('Support tokenizer has no supported chat template')
        tokenizer.chat_template = template
    for key in ['eos_token', 'bos_token', 'unk_token', 'pad_token']:
        if getattr(tokenizer, key) is None and isinstance(config.get(key), str):
            if config[key] not in tokenizer.get_vocab():
                raise ValueError('Configured special token missing from vocabulary')
            setattr(tokenizer, key, config[key])
    if tokenizer.eos_token_id is None:
        raise ValueError('Support tokenizer must define an EOS token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
