"""Redaction at the boundary; optional Langfuse spans, no raw callbacks."""
import os
import re
from contextlib import contextmanager
from time import perf_counter

def redact(value):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if re.search(r'password|secret|token|authorization|api.key', k, re.I) else redact(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'(?i)\b(bearer\s+)\S+', r'\1[REDACTED]', value)
        value = re.sub(r'(?i)\b(password|secret|token|api[_-]?key)\s*[:=]\s*[\"\']?[^\s,;\"\']+', r'\1=[REDACTED]', value)
        value = re.sub(r'\b(?:hf_|sk-)[A-Za-z0-9_-]{12,}', '[REDACTED]', value)
        value = re.sub(r'(://[^:/\s]+:)[^@\s]+@', r'\1[REDACTED]@', value)
    return value

class Observer:
    def __init__(self):
        self.client = None
        if os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
            from langfuse import Langfuse
            self.client = Langfuse()

    @contextmanager
    def span(self, name, inputs):
        started = perf_counter()
        record = {}
        cm = span = None
        if self.client:
            try:
                cm = self.client.start_as_current_observation(name=name, as_type='span', input=redact(inputs))
                span = cm.__enter__()
            except Exception:
                cm = None
        try:
            yield record
        except Exception as exc:
            record['error'] = type(exc).__name__
            raise
        finally:
            record['latency_seconds'] = perf_counter() - started
            if span:
                try:
                    span.update(output=redact(record))
                    cm.__exit__(None, None, None)
                except Exception:
                    pass  # Observability must not break a support request.

    def close(self):
        if self.client:
            self.client.flush()
