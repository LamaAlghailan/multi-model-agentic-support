from src.config import settings
from src.schemas import RoutingDecision
from src.models.intent_classifier import IntentClassifier


INTENT_TO_ROUTE = {
    "authentication": "qa",
    "network": "tools",
    "deployment": "support",
    "database": "qa",
    "gpu": "support",
    "api": "qa",
    "package": "tools",
    "general": "support",
}


class ClassifierRouter:
    def __init__(self):
        self.classifier = IntentClassifier()

    def route(
        self,
        user_message: str,
    ) -> RoutingDecision | None:

        intent, confidence = (
            self.classifier.predict(
                user_message
            )
        )

        if (
            confidence
            < settings.classifier_confidence_threshold
        ):
            return None

        route_name = INTENT_TO_ROUTE.get(
            intent,
            "support",
        )

        return RoutingDecision(
            route=route_name,
            intent=intent,
            confidence=confidence,
            reason=(
                "High-confidence Model A "
                "intent classification"
            ),
        )