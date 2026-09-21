import pytest
from src.routing.classifier_router import ClassifierRouter
from src.routing.llm_router import LLMRouter
from src.routing.hybrid_router import HybridRouter
from src.evaluation.metrics import threshold_sweep

class Classifier:
    def __init__(self,confidence=.95):self.confidence=confidence
    def predict(self,text):return 'database',self.confidence

class LLM:
    def route_json(self,text):return '{"route":"support","intent":"general","confidence":0.5,"reason":"ambiguous"}'

@pytest.mark.parametrize('text',['Production database corruption suspected','DELETE FROM tickets;', 'A production security breach'])
def test_safety_precedes_models(text):
    class Broken:
        def predict(self,text):raise AssertionError('should not call classifier')
        def route_json(self,text):raise AssertionError('should not call LLM')
    for router in [ClassifierRouter(Broken()),LLMRouter(Broken()),HybridRouter(Broken(),LLMRouter(Broken()),.8)]:
        assert router.route(text).route == 'escalate'

def test_model_a_confidence_regression():
    llm=LLMRouter(LLM())
    assert HybridRouter(Classifier(.207),llm,.8).route('pool busy').route == 'support'
    assert llm.calls == 1
    assert HybridRouter(Classifier(),llm,.8).route('pool busy').route == 'tools'
    assert llm.calls == 1

@pytest.mark.parametrize('raw',['not json','{}','{"route":"tools","intent":"database","confidence":2,"reason":"bad"}',
    '{"route":"tools","intent":"database","confidence":0.7,"reason":"bad","command":"drop"}'])
def test_llm_json_fallback(raw):
    class Invalid:
        def route_json(self,text):return raw
    router=LLMRouter(Invalid())
    assert router.route('hello').route=='escalate'
    assert router.invalid_json==1

def test_validation_policy_does_not_invent_coverage():
    result=threshold_sweep([0,1],[1,0],[.2,.3])
    assert not result['policy_qualified']
    assert result['selected_threshold']==1
