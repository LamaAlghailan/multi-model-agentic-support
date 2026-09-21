from src.config import LABELS

class IntentClassifier:
    def __init__(self, source):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(source, trust_remote_code=False).eval()
        if set(self.model.config.id2label.values()) != set(LABELS):
            raise ValueError('Model A label mapping does not match the intent taxonomy')

    def predict(self, text):
        inputs = self.tokenizer(text, truncation=True, max_length=128, return_tensors='pt')
        with self.torch.inference_mode():
            probs = self.model(**inputs).logits[0].softmax(-1)
        index = int(probs.argmax())
        return self.model.config.id2label[index], float(probs[index])
