import re
from src.schemas import RouteDecision

def safety_rule(text):
    t = text.lower()
    if (re.search(r'\b(delete\s+from|drop\s+(table|database)|truncate\s+table|rm\s+-rf)\b', t)
        or re.search(r'\b(corrupt\w*|data loss|security breach|ransomware|exfiltrat\w*)\b', t)
        or ('production' in t and re.search(r'disable\s+(authentication|auth|security)|deleted|down|outage', t))):
        return RouteDecision(route='escalate', intent='database' if re.search('database|sql|table', t) else 'general',
                             confidence=1, reason='High-risk incident or destructive action requires human review')
    return None

def intent_route(text, intent):
    if re.search(r'\b(docs?|documentation|guide|runbook|kb)\b', text.lower()):
        return 'qa'
    if intent == 'general' and not re.search(r'health|right now|logs|ticket|calculate', text.lower()):
        return 'support'
    return 'tools'
