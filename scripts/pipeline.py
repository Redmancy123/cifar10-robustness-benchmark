"""Reviewed unattended sequence: wait for verified data, benchmark, freeze, train, evaluate, plot."""
import csv
import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torchvision
from src.common import ROOT,canonical_hash,digest,now,source_fingerprints,write_json,verify_frozen
from src.runtime import run_guarded,terminate_owned


def status(phase,**kwargs):
    write_json(ROOT/'runs/pipeline_status.json',dict(time=now(),phase=phase,pid=os.getpid(),**kwargs))
    print(now(),phase,kwargs,flush=True)


def verify_gate(candidate_path):
    gate=json.loads((ROOT/'runs/review_gate.json').read_text(encoding='utf-8'))
    if not gate.get('approved') or gate['source_hashes']!=source_fingerprints():
        raise RuntimeError('Source review gate missing or source changed')
    if digest(candidate_path)!=gate['candidate_sha256']:
        raise RuntimeError('Candidate config changed after independent review')
    for file in ('cpu_checks.json','watchdog_checks.json','amp_health_checks.json'):
        if json.loads((ROOT/'runs'/file).read_text(encoding='utf-8'))['status']!='passed':
            raise RuntimeError('Required CPU checks not passed')


def main(a):
    candidate=Path(a.candidate).resolve()
    frozen=Path(a.frozen).resolve()
    if not candidate.is_relative_to(ROOT/'configs') or not frozen.is_relative_to(ROOT/'configs'):
        raise RuntimeError('Configuration paths must stay in project configs')
    verify_gate(candidate)
    started=time.monotonic()
    status('preparing_verified_data')
    if not (ROOT/'data/data_manifest.json').exists():
        import subprocess
        # Own preparation process; failure is visible and halts the pipeline, not a silent 24h wait.
        with (ROOT/'runs/download.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u','scripts/download_parallel.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            prep_start=time.monotonic()
            try:
                while child.poll() is None:
                    if (ROOT/'runs/STOP_ALL').exists() or time.monotonic()-prep_start>12*3600:
                        terminate_owned(child)
                        raise RuntimeError('Data preparation stopped or exceeded 12h')
                    time.sleep(5)
                if child.wait()!=0:raise RuntimeError('Data download/preparation failed; inspect runs/download.log')
            finally:
                if child.poll() is None:terminate_owned(child)
    cfg=json.loads(candidate.read_text(encoding='utf-8'))
    write_json(ROOT/'runs/environment.json',{'time':now(),'python':sys.executable,'platform':platform.platform(),
               'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,
               'cuda_runtime':torch.version.cuda,'gpu':'NVIDIA GeForce RTX 3050 Laptop GPU (verified existing environment)',
               'allocator_fraction':0.5,'allocator_fraction_is_not_total_cuda_memory_limit':True})
    if a.resume:
        cfg=json.loads(frozen.read_text(encoding='utf-8'));verify_frozen(cfg)
        chosen=cfg['epochs'];tau=cfg['estimated_epoch_seconds']
        status('resuming_frozen_matrix',config_hash=canonical_hash(cfg))
    else:
        if frozen.exists():raise RuntimeError('Frozen file exists; use explicit --resume after review')
        measured={}
        for method in cfg['methods']:
            verify_gate(candidate);status('benchmark',method=method)
            command=['scripts/train.py','--config',str(candidate),'--method',method,'--seed','101','--probe']
            if method in ('erm','augmix'):command+=['--probe-epochs','2']
            else:command+=['--probe-epochs','1','--max-batches','80']
            run_guarded(command,'probe_'+method,max_hours=2)
            file=ROOT/'runs/probes'/('%s_seed101_%s'%(method,canonical_hash(cfg)[:10]))/'benchmark.json'
            measured[method]=json.loads(file.read_text(encoding='utf-8'))
            if measured[method]['amp_skips']>5:raise RuntimeError('Probe AMP instability; requires investigation')
            if max(r['peak_allocated_mib'] for r in measured[method]['timings'])>1800:
                raise RuntimeError('Probe allocator peak leaves insufficient planned headroom')
        tau0=float(np.mean([r['epoch_s'] for r in measured['erm']['timings']]))
        tau2=float(np.mean([r['epoch_s'] for r in measured['augmix']['timings']]))
        tau={'erm':tau0,'mixup':tau0*1.05,'augmix':tau2,'warmup':tau2*1.03,'matched':tau2*1.03}
        val_s=max(r['val_s'] for m in measured.values() for r in m['timings'])
        startup_s=max(m['elapsed_s']-sum(r['epoch_s'] for r in m['timings']) for m in measured.values())
        eval_s=15*760000/5000*val_s*1.5+900
        estimates={e:(1.25*3*(sum(tau.values())*e+5*startup_s)+eval_s+900) for e in (100,80,60,40)}
        chosen=next((e for e in (100,80,60,40) if estimates[e]<=24*3600),None)
        write_json(ROOT/'runs/budget_estimate_v1_2.json',{'time':now(),'epoch_seconds':tau,'startup_s':startup_s,
                   'full_evaluation_estimate_s':eval_s,'total_estimates_hours':{str(e):s/3600 for e,s in estimates.items()},
                   'selected_epochs':chosen,'methods':cfg['methods'],'n_runs':15,'unmeasured_long_run_uncertainty':True})
        if chosen is None:raise RuntimeError('Even 40 epochs exceeds reviewed 24h reference budget; no formal training started')
        cfg.update(frozen=True,epochs=chosen,data_manifest_sha256=digest(ROOT/'data/data_manifest.json'),
                   source_hashes=source_fingerprints(),split_hashes={k:digest(ROOT/'data/splits'/(k+'_indices.npy')) for k in ('train','val')},
                   freeze_time=now(),epochs_reference='AugMix CIFAR default 100; MixUp and WRN reference 200; common resource-limited comparison',
                   budget_estimate_hours=estimates[chosen]/3600,estimated_epoch_seconds=tau)
        write_json(frozen,cfg)
    cfg_hash=canonical_hash(cfg)
    formal_start=time.monotonic();runs=[]
    for i,seed in enumerate(cfg['seeds']):
        methods=cfg['methods'][i:]+cfg['methods'][:i]
        for method in methods:
            if time.monotonic()-formal_start>24*3600:
                raise RuntimeError('Actual matrix wall time exceeded 24h review point; stop starting new runs')
            verify_gate(candidate);name='%s_seed%d_%s'%(method,seed,cfg_hash[:10]);runs.append(name)
            status('formal_training',method=method,seed=seed,epochs=chosen,planned_runs=15,started_runs=len(runs))
            run_guarded(['scripts/train.py','--config',str(frozen),'--method',method,'--seed',str(seed)],
                        'train_'+name,max_hours=min(8,max(2,3*chosen*tau[method]/3600)))
    for i,name in enumerate(runs):
        verify_gate(candidate);status('final_evaluation',run=name,index=i+1,total=len(runs))
        run_guarded(['scripts/evaluate.py','--run',str(ROOT/'runs/formal'/name)],'evaluate_'+name,max_hours=3)
    status('aggregating_and_plotting')
    import subprocess
    subprocess.run([sys.executable,'scripts/summarize.py','--config',str(frozen)],cwd=ROOT,check=True)
    status('completed',formal_wall_hours=(time.monotonic()-formal_start)/3600,total_wall_hours=(time.monotonic()-started)/3600,
           results=str(ROOT/'results'),n_runs=len(runs),epochs=chosen)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',default='configs/candidate_v1_2.json')
    parser.add_argument('--frozen',default='configs/frozen_v1_2.json')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    lock=ROOT/'runs/pipeline.lock'
    try:
        fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise SystemExit('Existing pipeline lock; inspect the owning run before retrying')
    with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:
        try:main(args)
        except BaseException as e:
            status('stopped_requires_review',error=repr(e))
            raise
    finally:
        lock.unlink(missing_ok=True)
