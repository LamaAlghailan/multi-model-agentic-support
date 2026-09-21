import json
from src.config import Settings, BASE_C
from src.graph import build_graph
from src.tools import ToolRegistry
from src.observability import Observer
from src.routing.llm_router import LLMRouter
from src.routing.hybrid_router import HybridRouter

class ObservedClassifier:
    def __init__(self, model, observer):
        self.model, self.observer = model, observer

    def predict(self, text):
        with self.observer.span('model_a', {'message': text}) as event:
            intent, confidence = self.model.predict(text)
            event.update(intent=intent, confidence=confidence)
            return intent, confidence

class ObservedRouterModel:
    def __init__(self, model, observer):
        self.model, self.observer = model, observer

    def route_json(self, text):
        with self.observer.span('model_c_router', {'message': text}) as event:
            output = self.model.route_json(text)
            event['output'] = output
            return output

def verify_release(settings):
    if not settings.release_file.exists():
        raise ValueError('Missing release evidence; run model evaluation and release checks')
    evidence = json.loads(settings.release_file.read_text(encoding='utf-8'))
    if evidence.get('eligible') is not True or not all(evidence.get('gates', {}).get(k) is True for k in ('model_a', 'model_b', 'model_c')):
        raise ValueError('Model quality gates have not all passed')
    expected = {'model_a': settings.model_a, 'model_b': settings.model_b, 'model_c': settings.model_c_adapter, 'base_c': BASE_C}
    if evidence.get('artifacts') != expected:
        raise ValueError('Release evidence does not match configured artifacts')
    from src.evaluation.release import artifact_digest
    if evidence.get('digests') != {k: artifact_digest(v) for k, v in expected.items() if k != 'base_c'}:
        raise ValueError('Model artifact contents differ from evaluated release')
    return float(evidence['threshold'])

def create_runtime(settings: Settings):
    observer = Observer()
    if settings.mode == 'demo':
        from src.demo import DemoClassifier, DemoQA, DemoSupport
        classifier, qa, support, threshold = DemoClassifier(), DemoQA(), DemoSupport(), 0.8
    else:
        threshold = verify_release(settings)
        from src.models.intent_classifier import IntentClassifier
        from src.models.qa_model import QAModel
        from src.models.support_model import SupportModel
        classifier = IntentClassifier(settings.model_a)
        qa = QAModel(settings.model_b)
        support = SupportModel(settings.model_c_adapter, settings.model_c_base)
    tools = ToolRegistry(settings.database, observer)
    router = HybridRouter(ObservedClassifier(classifier, observer), LLMRouter(ObservedRouterModel(support, observer)), threshold)
    return build_graph(router, qa, support, tools, observer), observer
