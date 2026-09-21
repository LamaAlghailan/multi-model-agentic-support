"""Fixed held-out routing cases; demo measurements never presented as model results."""
import argparse
import json
from time import perf_counter
from src.config import ROOT, Settings
from src.routing.classifier_router import ClassifierRouter
from src.routing.llm_router import LLMRouter
from src.routing.hybrid_router import HybridRouter
from src.evaluation.metrics import classification
from src.training.common import save

def main(demo=False):
    settings = Settings()
    if demo:
        from src.demo import DemoClassifier, DemoSupport
        classifier, support, threshold = DemoClassifier(), DemoSupport(), .8
    else:
        from src.models.intent_classifier import IntentClassifier
        from src.models.support_model import SupportModel
        classifier, support = IntentClassifier(settings.model_a), SupportModel(settings.model_c_adapter)
        threshold = json.loads((ROOT/'reports/threshold.json').read_text())['selected_threshold']
    routers = {'A':ClassifierRouter(classifier), 'B':LLMRouter(support), 'Hybrid':HybridRouter(classifier,LLMRouter(support),threshold)}
    cases = json.loads((ROOT/'data/router_test.json').read_text())
    results = {}
    for name, router in routers.items():
        details = []
        for case in cases:
            start = perf_counter(); decision = router.route(case['text']); elapsed = perf_counter()-start
            details.append({**case, 'prediction':decision.model_dump(), 'latency_ms':elapsed*1000, 'correct':decision.route == case['route']})
        metrics = classification([c['route'] for c in cases], [r['prediction']['route'] for r in details], ['qa','tools','support','escalate'])
        llm = router.llm_router if name == 'Hybrid' else router if name == 'B' else None
        results[name] = {**metrics, 'latency_ms_mean':sum(d['latency_ms'] for d in details)/len(details),
                         'llm_calls':llm.calls if llm else 0,
                         'invalid_json_rate':llm.invalid_json/llm.calls if llm and llm.calls else 0,
                         'llm_failures':llm.failures if llm else 0, 'details':details}
    report = {'mode':'demo_fixtures' if demo else 'real_models', 'threshold':threshold, 'results':results,
              'limitations':'Single warm-process run on a small fixed set; latency is environment-specific.'}
    save('router_demo.json' if demo else 'router_models.json', report)
    print(json.dumps({k:{m:v for m,v in r.items() if m not in ['details','confusion_matrix','per_class_recall']} for k,r in results.items()},indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--demo',action='store_true');args=parser.parse_args();main(args.demo)
