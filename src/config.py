from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_a_id: str = (
        "Lammem310/multi-model-support-intent-classifier"
    )

    classifier_confidence_threshold: float = 0.30

    llm_router_enabled: bool = True


settings = Settings()