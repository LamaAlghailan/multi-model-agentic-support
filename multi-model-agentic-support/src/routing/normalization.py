"""Explicit aliases only; unknown labels remain invalid, never guessed."""
from src.config import LABELS

ALIASES = {'auth': 'authentication', 'authentication_issue': 'authentication',
           'network_issue': 'network', 'deployment_issue': 'deployment',
           'db': 'database', 'database_issue': 'database', 'cuda': 'gpu',
           'gpu_issue': 'gpu', 'api_issue': 'api', 'dependency': 'package',
           'package_issue': 'package', 'general_question': 'general'}

def normalize_intent(value):
    if not isinstance(value, str):
        raise ValueError('Intent must be a string')
    key = value.strip().lower().replace('-', '_').replace(' ', '_')
    key = ALIASES.get(key, key)
    if key not in LABELS:
        raise ValueError('Unknown intent')
    return key
