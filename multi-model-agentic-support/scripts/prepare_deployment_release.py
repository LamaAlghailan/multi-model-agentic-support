"""Relocate an already passing release without discarding artifact-hash verification."""
import json
from src.config import ROOT
from src.evaluation.release import artifact_digest
from src.training.common import save

def main():
    release=json.loads((ROOT/'reports/release.json').read_text())
    if release.get('eligible') is not True: raise ValueError('Original release is not eligible')
    paths={'model_a':ROOT/'models/intent_classifier','model_b':ROOT/'models/qa_model','model_c':ROOT/'models/support_adapter'}
    for key,path in paths.items():
        if artifact_digest(str(path)) != release['digests'][key]:
            raise ValueError('Transferred artifact hash mismatch: '+key)
    for key,path in paths.items():release['artifacts'][key]=str(path)
    save('release.json',release)
    print('Verified model hashes and relocated release paths.')

if __name__=='__main__':main()
