"""Read-only checkpoint/metric integrity checks; never construct or run a model."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.dataset import split_fingerprint
from src.lofo import FAMILIES,ORIGINAL_SPLIT_SHA256
from src.utils import load_checkpoint
from src.controlled_v2.provenance import load_freeze
from src.controlled_v2.train import verify_best


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def source_hash(folder):
    digest=hashlib.sha256()
    for path in sorted(folder.glob('*.py')):
        digest.update(path.name.encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def verify():
    import os
    os.chdir(ROOT)
    torch.set_num_threads(4)
    split=read(ROOT/'splits/cifar100_seed42.json')
    assert split_fingerprint(split)==ORIGINAL_SPLIT_SHA256
    freeze=load_freeze()
    legacy_manifest=read(ROOT/'results/evaluation/seed42/manifests/provenance.json')
    bridge_manifest=read(ROOT/'results/evaluation_controlled_v2/seed42/manifest.json')
    legacy_names=('clean_baseline',*(f'lofo_{f}' for f in FAMILIES))
    runs=[('historical',name,ROOT/'results'/name/('clean_seed42' if name=='clean_baseline' else 'seed42')) for name in legacy_names]
    runs += [('controlled_v2',name,ROOT/'results/controlled_v2'/name/'seed42') for name in ('clean_baseline','all_four_representatives')]
    summaries=[]
    for cohort,name,run in runs:
        best_path,last_path=(run/'checkpoints'/p for p in ('best_model.pt','last_checkpoint.pt'))
        best,last=load_checkpoint(best_path),load_checkpoint(last_path)
        history=last['history']; exported=read(run/'metrics.json')
        assert history==exported and len(history)==100,(cohort,name,'history')
        assert [r['epoch'] for r in history]==list(range(1,101))
        selected=max(history,key=lambda r:r['val_accuracy_pct'])
        assert best['epoch']==last['best_epoch']==selected['epoch']
        assert best['best_validation_accuracy_pct']==selected['val_accuracy_pct']
        assert best['history']==history[:best['epoch']]
        assert last['scheduler_state_dict']['last_epoch']==100
        if cohort=='historical':
            assert read(run/'config.json')==best['config']==last['config']
            assert source_hash(run/'source')==best['source_sha256']==last['source_sha256']
            data=last['data_manifest']
            assert sha(best_path)==legacy_manifest['checkpoints'][name]['sha256']
        else:
            manifest=read(run/'run_manifest.json')
            verify_best(best,last,manifest,complete=True)
            assert manifest['freeze_sha256']==freeze['freeze_sha256']
            data=manifest['data_manifest']
            info=bridge_manifest['checkpoints'][name]
            assert sha(best_path)==info['best_sha256'] and sha(last_path)==info['last_sha256']
        assert data['split_sha256']==ORIGINAL_SPLIT_SHA256
        assert data['train_indices']==split['train_indices'] and data['val_indices']==split['val_indices']
        assert data['train_count']==45000 and data['val_count']==5000 and len(data['classes'])==100
        for checkpoint in (best,last):
            for tensor in checkpoint['model_state_dict'].values():
                assert not tensor.is_floating_point() or (tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all()))
        with (run/'metrics.csv').open() as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==100
        for row,record in zip(rows,history):
            for key in ('epoch','train_loss','val_loss','train_accuracy_pct','val_accuracy_pct','learning_rate'):
                assert math.isclose(float(row[key]),float(record[key]),abs_tol=1e-12,rel_tol=0)
        summaries.append({'cohort':cohort,'model':name,'epochs':100,'selected_epoch':best['epoch'],
                          'best_clean_validation_accuracy_pct':selected['val_accuracy_pct'],
                          'best_checkpoint_sha256':sha(best_path),'last_checkpoint_sha256':sha(last_path)})
    evaluations=[]
    for relative,expected,names in (('results/evaluation/seed42',380,set(legacy_names)),
                                    ('results/evaluation_controlled_v2/seed42',152,{'clean_baseline','all_four_representatives'})):
        folder=ROOT/relative
        result=read(folder/'metrics/evaluation.json');conditions=result['conditions']
        assert len(conditions)==expected and len({(r['model'],r['corruption'],r['severity']) for r in conditions})==expected
        assert {r['model'] for r in conditions}==names
        assert all(r['n_samples']==10000 and r['seed']==42 for r in conditions)
        assert all(math.isclose(r['accuracy_pct'],r['n_correct']/100,abs_tol=1e-10,rel_tol=0) for r in conditions)
        assert len(list((folder/'cache').rglob('*.npz')))==expected
        if 'controlled_v2' in relative:assert read(folder/'status.json')['status']=='complete'
        else:assert result['summary']['benchmark_complete']
        evaluations.append({'path':relative,'condition_rows':expected,'prediction_caches':expected,'status':'complete'})
    return {'models_verified':len(summaries),'runs':summaries,'evaluations':evaluations,
            'frozen_v2_valid':True,'model_forward_passes':0,'training_or_inference_run':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',help='Optional new verification JSON; existing results are never changed')
    args=parser.parse_args()
    report=verify()
    text=json.dumps(report,indent=2)+'\n'
    if args.output:
        path=Path(args.output)
        if path.exists():raise FileExistsError('Verification report exists; use a new output path')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    print(text,end='')
