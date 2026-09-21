"""Central model identity and environment configuration; no embedded credentials."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('authentication', 'network', 'deployment', 'database', 'gpu', 'api', 'package', 'general')
MODEL_NAME = 'tuwaiq-tech-support-agent'
BASE_AB = 'distilbert-base-uncased'
BASE_C = 'HuggingFaceTB/SmolLM2-135M-Instruct'
HUB_A = 'Lammem310/multi-model-support-intent-classifier'
HUB_B = 'Lammem310/multi-model-support-extractive-qa'
# Historical adapter uses 360M and MUST NOT be loaded into the 135M base.
HISTORICAL_C = 'Lammem310/multi-model-support-specialist-lora'

def model_source(env, local, fallback):
    return os.getenv(env) or (str(ROOT / local) if (ROOT / local / 'config.json').exists() else fallback)

@dataclass
class Settings:
    mode: str = field(default_factory=lambda: os.getenv('AGENT_MODE', 'production'))
    model_a: str = field(default_factory=lambda: model_source('MODEL_A_PATH', 'models/intent_classifier', HUB_A))
    model_b: str = field(default_factory=lambda: model_source('MODEL_B_PATH', 'models/qa_model', HUB_B))
    model_c_base: str = BASE_C
    model_c_adapter: str = field(default_factory=lambda: os.getenv('MODEL_C_ADAPTER', str(ROOT / 'models/support_adapter')))
    release_file: Path = field(default_factory=lambda: Path(os.getenv('RELEASE_FILE', str(ROOT / 'reports/release.json'))))
    database: Path = field(default_factory=lambda: Path(os.getenv('SUPPORT_DB', str(ROOT / 'runtime/support.db'))))
    api_key: str = field(default_factory=lambda: os.getenv('AGENT_API_KEY', ''))
    max_concurrency: int = 1

    def __post_init__(self):
        if self.mode not in {'production', 'demo'}:
            raise ValueError('AGENT_MODE must be production or demo')
