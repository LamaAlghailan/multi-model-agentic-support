import pytest
from src.tools.registry import CONTRACTS
from src.schemas import ToolResult

def test_all_contracts_have_implementations(tools):
    assert len(CONTRACTS) == 13
    assert all(callable(getattr(tools,k)) for k in CONTRACTS)

def test_ticket_round_trip_and_safe_parameters(tools):
    result = tools.invoke('ticket_create', {'title':"O'Reilly issue",'description':'password=topsecret API fails'})
    assert result['ok'] and result['data']['id'] == 1
    found = tools.invoke('ticket_search',{'query':"O'Reilly"})
    assert len(found['data']['tickets']) == 1
    assert 'topsecret' not in str(found)
    assert tools.invoke('sql_query',{'query':'SELECT count(*) AS n FROM tickets'})['data']['rows'][0]['n'] == 1

@pytest.mark.parametrize('query', ['DELETE FROM tickets', 'DROP TABLE tickets', 'SELECT 1; DELETE FROM tickets',
    "SELECT load_extension('anything')", "ATTACH DATABASE 'other.db' AS other", 'PRAGMA user_version=2',
    'WITH n AS (SELECT 1) DELETE FROM tickets', 'SELECT * FROM escalations'])
def test_sql_is_enforced_read_only(tools,query):
    assert not tools.invoke('sql_query',{'query':query})['ok']

def test_recursive_query_has_budget(tools):
    r=tools.invoke('sql_query',{'query':'WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n'})
    assert not r['ok']

@pytest.mark.parametrize('expression',['__import__("os").getcwd()', '2**999999', '1/0', '[1,2]', 'True+1'])
def test_calculator_rejects_unsafe_or_invalid(tools,expression):
    assert not tools.invoke('calculator',{'expression':expression})['ok']

def test_structured_functional_tools(tools):
    cases = {'knowledge_base_search':{'query':'application port'}, 'documentation_search':{'query':'health'},
             'system_health_check':{'service':'database'}, 'log_analyzer':{'text':'CUDA out of memory'},
             'package_lookup':{'name':'pydantic'}, 'calculator':{'expression':'(2+3)*4'},
             'file_search':{'query':'api'}, 'diagnostic_runbook':{'intent':'database'},
             'escalate_to_human':{'reason':'corruption','evidence':'logs show failed migration'}}
    for name, args in cases.items():
        result = ToolResult.model_validate(tools.invoke(name,args))
        assert result.ok, result
    assert tools.invoke('calculator',{'expression':'(2+3)*4'})['data']['value'] == 20
    assert tools.invoke('system_health_check',{'service':'database'})['data']['is_live'] is False
    assert not tools.invoke('web_search',{'query':'latest package'})['ok']

def test_contract_errors(tools):
    for name,args in [('missing',{}),('system_health_check',{'service':'secret_host'}),('calculator',{'expression':'1','extra':True})]:
        assert not tools.invoke(name,args)['ok']
