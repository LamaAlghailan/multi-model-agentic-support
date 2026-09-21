from src.schemas import RouteDecision
from .rules import safety_rule

class LLMRouter:
    def __init__(self, model):
        self.model = model
        self.calls = self.invalid_json = self.failures = 0

    def route(self, text):
        if decision := safety_rule(text):
            return decision
        self.calls += 1
        try:
            raw = self.model.route_json(text)
        except Exception:
            self.failures += 1
            return self.fallback('LLM router unavailable')
        try:
            return RouteDecision.model_validate_json(raw, strict=True)
        except (ValueError, TypeError):
            self.invalid_json += 1
            return self.fallback('LLM router returned invalid JSON')

    @staticmethod
    def fallback(reason):
        return RouteDecision(route='escalate', intent='general', confidence=0, reason=reason)
