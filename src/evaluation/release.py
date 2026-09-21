"""Produce a fail-closed, artifact-bound release record from measured reports."""
import hashlib
import json
from pathlib import Path
from src.config import ROOT, BASE_C
from src.training.common import save

def artifact_digest(source):
    root = Path(source)
    if not root.is_dir():
        raise ValueError('Release requires local immutable model snapshots; download Hub models first')
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('Missing model weights')
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def main():
    reports = {}
    for key in ['model_a','model_b','model_c']:
        path = ROOT/'reports'/f'{key}.json'
        reports[key] = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    a,b,c = (reports[k] for k in ['model_a','model_b','model_c'])
    gates = {'model_a': a.get('f1_macro',0) >= .8 and min(a.get('per_class_recall',{'missing':0}).values()) >= .6 and a.get('reload_ok') is True,
             'model_b': b.get('exact_match',0) >= .65 and b.get('token_f1',0) >= .8 and b.get('reload_ok') is True,
             'model_c': c.get('base') == BASE_C and c.get('assistant_only') is True and c.get('golden',{}).get('pass_rate') == 1
                        and c.get('loss',float('inf')) < c.get('baseline_loss',0) and c.get('reload_ok') is True}
    artifacts = {k:r.get('artifact','') for k,r in reports.items()}
    artifacts['base_c'] = BASE_C
    result = {'eligible':all(gates.values()), 'gates':gates, 'artifacts':artifacts,
              'threshold':a.get('threshold',{}).get('selected_threshold',1.0)}
    if result['eligible']:
        result['digests'] = {k:artifact_digest(v) for k,v in artifacts.items() if k != 'base_c'}
    save('release.json', result)
    print(json.dumps(result,indent=2))
    return result

if __name__ == '__main__':
    raise SystemExit(0 if main()['eligible'] else 1)
