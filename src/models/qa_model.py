"""Extractive QA with context-only spans across all overflow windows."""

def best_span(context, offsets, starts, ends, max_answer_length=40):
    candidates = []
    valid = [i for i, offset in enumerate(offsets) if offset and offset[1] > offset[0]]
    for i in sorted(valid, key=lambda i: starts[i], reverse=True)[:20]:
        for j in sorted(valid, key=lambda j: ends[j], reverse=True)[:20]:
            if i <= j < i + max_answer_length and all(offsets[k] for k in range(i, j + 1)):
                candidates.append((float(starts[i] + ends[j]), offsets[i][0], offsets[j][1]))
    if not candidates:
        return {'text': '', 'score': float('-inf'), 'start': 0, 'end': 0}
    score, start, end = max(candidates)
    return {'text': context[start:end], 'score': score, 'start': start, 'end': end}

class QAModel:
    def __init__(self, source):
        import torch
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
        self.model = AutoModelForQuestionAnswering.from_pretrained(source, trust_remote_code=False).eval()

    def answer(self, question, context):
        if not context:
            return {'text': '', 'start': 0, 'end': 0, 'windows': 0}
        batch = self.tokenizer(question.strip(), context, truncation='only_second', max_length=384,
                               stride=96, return_overflowing_tokens=True, return_offsets_mapping=True,
                               padding=True, return_tensors='pt')
        offsets = batch.pop('offset_mapping').tolist()
        batch.pop('overflow_to_sample_mapping')
        spans = []
        # Bound peak memory for arbitrarily long retrieved documents.
        for i in range(len(offsets)):
            with self.torch.inference_mode():
                out = self.model(**{k: v[i:i+1] for k, v in batch.items()})
            context_offsets = [o if s == 1 else None for o, s in zip(offsets[i], batch.sequence_ids(i))]
            spans.append(best_span(context, context_offsets, out.start_logits[0].tolist(), out.end_logits[0].tolist()))
        best = max(spans, key=lambda s: s['score'])
        best['windows'] = len(spans)
        return best
