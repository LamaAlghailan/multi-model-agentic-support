import json
import pytest
from fastapi.testclient import TestClient
from src.api import create_app
from src.config import Settings, MODEL_NAME
from src.demo import DemoClassifier, DemoQA, DemoSupport
from src.graph import build_graph
from src.observability import Observer
from src.routing.classifier_router import ClassifierRouter
from src.routing.hybrid_router import HybridRouter
from src.routing.llm_router import LLMRouter
from src.tools.selector import select_tools

@pytest.mark.parametrize('alias,expected', [('DB','database'),('auth','authentication'),('GPU issue','gpu'),('general-question','general')])
def test_aliases_all_routers(alias, expected):
    class Classifier:
        def predict(self, text): return alias, .9
    class Model:
        def route_json(self, text): return json.dumps(dict(route='support',intent=alias,confidence=.7,reason='test'))
    llm=LLMRouter(Model())
    for router in [ClassifierRouter(Classifier()), llm, HybridRouter(Classifier(),llm,.8)]:
        assert router.route('help').intent == expected

@pytest.mark.parametrize('value', ['unknown', None, 3])
def test_unknown_intents_fail_closed(value):
    class Model:
        def predict(self,text): return value,.99
        def route_json(self,text): return json.dumps(dict(route='tools',intent=value,confidence=.9,reason='test'))
    assert ClassifierRouter(Model()).route('help').route == 'escalate'
    assert HybridRouter(Model(), LLMRouter(Model()),.8).route('help').route == 'escalate'

@pytest.mark.parametrize('message,intent,tool', [('calculate (2+3)*4','general','calculator'),('package pydantic','package','package_lookup'),('find tickets outage','general','ticket_search'),('pool exhausted','database','system_health_check')])
def test_selector(message,intent,tool):
    assert select_tools(message,intent)[0][0] == tool
    assert all(name not in {'sql_query','ticket_create'} for name,args in select_tools(message,intent))

@pytest.mark.parametrize('branch',['router','qa','support','tools','escalation'])
def test_graph_failure_fallback(tools,branch):
    router=HybridRouter(DemoClassifier(),LLMRouter(DemoSupport()),.8)
    qa,support=DemoQA(),DemoSupport()
    def fail(*args,**kwargs): raise RuntimeError('private exception details')
    message='Explain LoRA'
    if branch=='router': router.route=fail
    if branch=='qa':
        qa.answer=fail
        message='According to the deployment guide which port?'
    if branch=='support': support.answer=fail
    if branch=='tools':
        tools.system_health_check=fail
        message='database pool exhausted'
    if branch=='escalation':
        tools.escalate_to_human=fail
        message='production database corruption'
    result=build_graph(router,qa,support,tools,Observer()).invoke({'user_message':message,'trace_id':'test'})
    assert result['escalate'] and 'Human review required' in result['answer']
    assert 'private exception' not in str(result)


def test_api_escalation_and_calculation(tmp_path):
    with TestClient(create_app(Settings(mode='demo',database=tmp_path/'test.db'))) as client:
        assert client.get('/health').status_code == 200
        assert [item['id'] for item in client.get('/v1/models').json()['data']] == [MODEL_NAME]
        for message,route in [('production database corruption','escalate'),('calculate 2+3','tools')]:
            response=client.post('/v1/chat/completions',json={'model':MODEL_NAME,'messages':[{'role':'user','content':message}]})
            assert response.status_code==200
            data=response.json()
            assert data['object']=='chat.completion' and data['model']==MODEL_NAME
            assert data['system_metadata']['route']==route
            assert data['choices'][0]['finish_reason']=='stop'
            if route=='escalate': assert 'no human has been notified' in data['choices'][0]['message']['content']
            else: assert '"value": 5' in data['choices'][0]['message']['content']


def test_graph_input_schema_is_valid(tools):
    graph = build_graph(HybridRouter(DemoClassifier(), LLMRouter(DemoSupport()), .8),
                        DemoQA(), DemoSupport(), tools, Observer())
    schema = graph.get_input_schema().model_json_schema()
    if '$ref' in schema:
        schema = schema['$defs'][schema['$ref'].rsplit('/', 1)[-1]]
    assert 'user_message' in schema['properties']


@pytest.mark.parametrize('failure', ['qa_model', 'second_tool', 'qa_tool'])
def test_failure_preserves_diagnostic_evidence(tools, failure):
    qa = DemoQA()
    def fail(*args, **kwargs):
        raise RuntimeError('private failure text')
    message = 'According to the deployment guide which port?'
    expected = 'knowledge_base_search'
    if failure == 'qa_model':
        qa.answer = fail
    elif failure == 'qa_tool':
        tools.knowledge_base_search = lambda **kwargs: (_ for _ in ()).throw(ValueError('search unavailable'))
    else:
        message = 'database pool exhausted'
        tools.log_analyzer = fail
        expected = 'system_health_check'
    graph = build_graph(HybridRouter(DemoClassifier(), LLMRouter(DemoSupport()), .8),
                        qa, DemoSupport(), tools, Observer())
    result = graph.invoke({'user_message': message, 'trace_id': 'evidence-test'})
    assert result['escalate'] and result['answer']
    assert any(item['tool'] == expected for item in result['tool_results'])
    assert 'private failure text' not in str(result)
