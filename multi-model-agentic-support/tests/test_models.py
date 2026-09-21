import json
import pytest
from src.config import ROOT
from src.models.qa_model import best_span
from src.evaluation.metrics import qa_scores,rouge_l
from src.evaluation.golden import check
from src.training.assistant_loss import assistant_features

def test_model_b_context_only_span_regression():
    # Question token has highest logits and must not be selected.
    result=best_span('port 8000', [None,(0,4),(5,9),None], [100,1,5,99],[100,1,8,99])
    assert result['text']=='8000' and result['start']==5

def test_qa_literal_spans_and_document_splits():
    from src.training.common import split_qa
    rows=json.loads((ROOT/'data/qa.json').read_text())
    assert len(rows)>=30
    assert all(r['context'][r['answer_start']:r['answer_start']+len(r['answer_text'])]==r['answer_text'] for r in rows)
    a,b,c=split_qa(rows)
    assert not ({r['doc_id'] for r in a}&{r['doc_id'] for r in b+c})
    assert all(len(r['context'])>2000 for r in c)

def test_scores_handle_repeated_tokens():
    assert qa_scores('the port','port')['exact_match']==1
    assert qa_scores('a a x x','x')['token_f1']==pytest.approx(2/3)
    assert rouge_l('a b c','a b c')==1

def test_model_c_golden_evaluator_regressions():
    assert check('G04','1. Inspect logs.\n2. Reproduce.\n3. Check dependencies.')
    assert not check('G04','1. Inspect logs.\n2. Reproduce.')
    assert check('G07','No, you should not run DELETE FROM tickets to test the database.')
    assert not check('G07','Yes, run DELETE FROM tickets now.')
    assert not check('G08','Yes.')

class CharacterTokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        text=''.join(m['role']+':'+m['content']+'|' for m in messages)
        return text+('assistant:' if add_generation_prompt else '')
    def __call__(self,text,**kwargs):return {'input_ids':[ord(c) for c in text]}

def test_assistant_only_labels():
    messages=[{'role':'system','content':'rules'},{'role':'user','content':'question'},{'role':'assistant','content':'answer'}]
    out=assistant_features(CharacterTokenizer(),messages)
    visible=''.join(chr(x) for x in out['labels'] if x!=-100)
    assert visible=='answer|'
    with pytest.raises(ValueError):assistant_features(CharacterTokenizer(),messages,max_length=5)

@pytest.mark.integration
def test_real_model_release_evidence():
    import os
    if os.getenv('RUN_MODEL_TESTS')!='1': pytest.skip('Set RUN_MODEL_TESTS=1 after running real model evaluations')
    from src.evaluation.release import main
    assert main()['eligible'], 'Real model gates failed; never replace this check with demo results'
