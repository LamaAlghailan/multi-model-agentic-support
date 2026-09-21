from .normalization import normalize_intent
from src.schemas import RouteDecision
from .rules import safety_rule, intent_route

class ClassifierRouter:
    def __init__(self, classifier):
        self.classifier = classifier

    def route(self, text):
        if decision := safety_rule(text):
            return decision
        try:
            intent, confidence = self.classifier.predict(text)
            intent = normalize_intent(intent)
            return RouteDecision(route=intent_route(text, intent), intent=intent, confidence=confidence, reason='Model A classification')
        except Exception:
            return RouteDecision(route='escalate', intent='general', confidence=0, reason='Classifier unavailable or invalid output')
