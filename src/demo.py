"""Deterministic demo fixtures. These are never used for model quality measurements."""
import json
from src.routing.rules import intent_route

class DemoClassifier:
    def predict(self, text):
        mapping = {'authentication': ['login', 'mfa', 'token'], 'network': ['vpn', 'dns', 'network'],
                   'deployment': ['docker', 'deployment', 'container', 'port'], 'database': ['database', 'sql', 'pool'],
                   'gpu': ['gpu', 'cuda'], 'api': ['api', '422', '503'], 'package': ['package', 'pip', 'version']}
        for intent, words in mapping.items():
            if any(w in text.lower() for w in words):
                return intent, 0.9
        return 'general', 0.4

class DemoQA:
    def answer(self, question, context):
        return {'text': context.split('. ')[0], 'windows': 1}

class DemoSupport:
    def answer(self, message, context='', tool_results=None, history=None, max_tokens=180):
        if tool_results:
            return 'DEMO: Diagnostic results (mock health is not live telemetry): ' + json.dumps(tool_results)
        return 'DEMO: Collect relevant logs, configuration and health evidence before concluding the cause.'

    def route_json(self, text):
        intent, _ = DemoClassifier().predict(text)
        return json.dumps({'route': intent_route(text, intent), 'intent': intent, 'confidence': 0.5, 'reason': 'Demo fixture'})
