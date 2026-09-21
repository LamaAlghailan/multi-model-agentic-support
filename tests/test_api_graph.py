import pytest
from fastapi.testclient import TestClient
from src.api import create_app
from src.config import Settings,MODEL_NAME
from src.runtime import create_runtime
from src.observability import redact

@pytest.mark.parametrize('text,route,path',[
    ('Production database corruption suspected','escalate',['route_node','escalation_node']),
    ('According to the deployment guide which port?','qa',['route_node','qa_node']),
    ('The database pool is exhausted','tools',['route_node','tools_node','support_node']),
    ('Explain LoRA','support',['route_node','support_node'])])
def test_langgraph_paths(tmp_path,text,route,path):
    graph,observer=create_runtime(Settings(mode='demo',database=tmp_path/'support.db'))
    result=graph.invoke({'user_message':text,'trace_id':'test','tool_results':[],'context':''})
    assert result['route']==route and result['path']==path and result['answer']

def test_api_contract_and_auth(tmp_path):
    with TestClient(create_app(Settings(mode='demo',database=tmp_path/'support.db',api_key='test-key'))) as client:
        assert client.get('/health').json()['mode']=='demo'
        assert client.get('/v1/models').status_code==401
        headers={'Authorization':'Bearer test-key'}
        assert [m['id'] for m in client.get('/v1/models',headers=headers).json()['data']]==[MODEL_NAME]
        body={'model':MODEL_NAME,'messages':[{'role':'user','content':'Explain LoRA'}]}
        result=client.post('/v1/chat/completions',headers=headers,json=body)
        assert result.status_code==200
        assert result.json()['choices'][0]['message']['role']=='assistant'
        assert result.json()['system_metadata']['mode']=='demo'
        for patch,status in [({'stream':True},400),({'model':'other'},404),({'messages':[]},422)]:
            response=client.post('/v1/chat/completions',headers=headers,json={**body,**patch})
            assert response.status_code==status and 'error' in response.json()

def test_production_missing_evidence_is_not_ready(tmp_path):
    with TestClient(create_app(Settings(mode='production',release_file=tmp_path/'absent.json'))) as client:
        assert client.get('/health').status_code==503
        assert client.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'hi'}]}).status_code==503

def test_redaction():
    value=redact({'password':'secret','message':'Authorization: Bearer abc123 password=hunter2 token=abc', 'url':'postgresql://user:password@host/db'})
    assert 'hunter2' not in str(value) and 'abc123' not in str(value) and ':password@' not in str(value)
