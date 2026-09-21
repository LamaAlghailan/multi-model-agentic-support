from .normalization import normalize_intent
from .rules import safety_rule, intent_route
from src.schemas import RouteDecision

class HybridRouter:
    def __init__(self, classifier, llm_router, threshold):
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold outside [0,1]')
        self.classifier, self.llm_router, self.threshold = classifier, llm_router, threshold

    def route(self, text):
        if decision := safety_rule(text):
            return decision
        try:
            intent, confidence = self.classifier.predict(text)
            intent = normalize_intent(intent)
            if confidence >= self.threshold:
                return RouteDecision(route=intent_route(text, intent), intent=intent, confidence=confidence,
                                     reason='Model A confidence meets validation-selected threshold')
        except Exception:
            pass
        return self.llm_router.route(text)
