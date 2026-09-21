"""Real-tokenizer regression checks without training or adjusting quality gates."""
import json
from src.config import BASE_AB, BASE_C
from src.models.tokenizer import load_support_tokenizer
from src.training.assistant_loss import assistant_features, AssistantCollator
from src.training.train_b import features
from src.training.common import read,save

def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(BASE_AB)
    rows=[]
    for position in [0,250,330,380,600,900]:
        context=' '.join(['routine']*position+['TARGET','VALUE']+['check']*200)
        rows.append({'question':'Which value is documented?','context':context,'answer_text':'TARGET VALUE','answer_start':context.index('TARGET')})
    batch,metadata=features(tok,rows)
    findings=[]
    for sample,row in enumerate(rows):
        windows=[i for i,m in enumerate(metadata) if m['sample']==sample]
        positive=[]
        for i in windows:
            start,end=batch['start_positions'][i],batch['end_positions'][i]
            if metadata[i]['offsets'][start] is not None:
                offsets=metadata[i]['offsets']
                span=row['context'][offsets[start][0]:offsets[end][1]]
                assert span=='TARGET VALUE'
                positive.append(i)
        assert positive
        findings.append({'answer_char_start':row['answer_start'],'windows':len(windows),'positive_windows':len(positive)})
    tokenizer=load_support_tokenizer(BASE_C)
    examples=read('sft_original.json')+read('sft_contrastive.json')
    encoded=[assistant_features(tokenizer,r['messages']) for r in examples]
    collated=AssistantCollator(tokenizer)(encoded[:4])
    assert (collated['labels'][collated['attention_mask']==0]==-100).all()
    assert all(any(x==-100 for x in r['labels']) and any(x!=-100 for x in r['labels']) for r in encoded)
    report={'qa_boundary_checks':findings,'qa_gate':True,'assistant_only_conversations_checked':len(encoded),
            'assistant_mask_gate':True,'max_conversation_tokens':max(len(r['input_ids']) for r in encoded)}
    save('preprocessing_checks.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
