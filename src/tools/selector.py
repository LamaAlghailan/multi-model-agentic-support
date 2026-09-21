"""Deterministic, read-only tool selection; no inferred writes or SQL."""
import re

def select_tools(message, intent):
    calls = []
    arithmetic = re.fullmatch(r'\s*calculate\s+([0-9.()+*/\s-]+)\s*[?]?\s*', message, re.I)
    package = re.fullmatch(r'\s*(?:package|version of)\s+([a-zA-Z0-9_.-]{1,80})\s*[?]?\s*', message, re.I)
    if arithmetic:
        return [('calculator', {'expression': arithmetic[1].strip()})]
    if package:
        return [('package_lookup', {'name': package[1]})]
    if re.search(r'\b(?:find|search)\s+tickets?\b', message, re.I):
        query = re.split(r'\b(?:find|search)\s+tickets?\b', message, flags=re.I)[-1].strip(' :')
        if query:
            return [('ticket_search', {'query': query[:2000]})]
    if intent in {'database', 'api', 'gpu', 'deployment', 'network'}:
        calls.append(('system_health_check', {'service': intent}))
    calls.extend([('log_analyzer', {'text': message}), ('diagnostic_runbook', {'intent': intent})])
    return calls
