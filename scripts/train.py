"""Training reads only the official training pool and frozen train/validation indices."""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from src.common import (ROOT, METHODS, canonical_hash, configure_torch, digest, now,
                        safety_checkpoint, save_torch, seed_all, write_json, verify_frozen,source_fingerprints)
from src.data import TrainDataset, EvalDataset, load_train_pool
from src.model import WideResNet
from src.metrics import Metrics, jsd_loss, beta_value, lr_factor
from src.amp_health import AMPHealth


@torch.inference_mode()
def validate(model, loader, amp):
    model.eval()
    metrics = Metrics()
    for x, y in loader:
        safety_checkpoint()
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            logits = model(x)
        metrics.add(logits, y)
    return metrics.result()


def main(a):
    config = json.loads(Path(a.config).read_text(encoding='utf-8'))
    if a.method not in METHODS:
        raise ValueError(a.method)
    if not a.probe:
        verify_frozen(config)
        if a.seed not in config['seeds'] or a.method not in config['methods']:
            raise RuntimeError('Undeclared formal method or seed')
    configure_torch()
    safety_checkpoint()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; will not silently fall back to a slow CPU run')
    seed_all(a.seed)
    manifest = json.loads((ROOT/'data/data_manifest.json').read_text(encoding='utf-8'))
    for kind in ('train', 'val'):
        if digest(ROOT/'data/splits'/('%s_indices.npy' % kind)) != manifest[kind+'_indices_sha256']:
            raise RuntimeError('Split indices changed')
    cfg_hash = canonical_hash(config)
    run_dir = ROOT / ('runs/probes' if a.probe else 'runs/formal') / ('%s_seed%d_%s' % (a.method,a.seed,cfg_hash[:10]))
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir/'completed.json').exists():
        completed=json.loads((run_dir/'completed.json').read_text(encoding='utf-8'))
        if digest(run_dir/'best.pt') != completed['best_sha256']:
            raise RuntimeError('Completed checkpoint changed')
        print('Already completed:',run_dir,flush=True)
        return
    run_config = dict(config, method=a.method, seed=a.seed, config_hash=cfg_hash, probe=a.probe,
                      actual_source_hashes=source_fingerprints())
    write_json(run_dir/'run_config.json', run_config)
    started = time.monotonic()
    print('Loading verified training/validation pool',flush=True)
    images, labels = load_train_pool()
    train_ids = np.load(ROOT/'data/splits/train_indices.npy')
    val_ids = np.load(ROOT/'data/splits/val_indices.npy')
    dataset = TrainDataset(images, labels, train_ids, a.method, a.seed)
    sampler_gen = torch.Generator()
    sampler = RandomSampler(dataset,generator=sampler_gen)
    loader_args = dict(batch_size=config['batch_size'], sampler=sampler, num_workers=config['workers'],
                       pin_memory=True, persistent_workers=config['workers'] > 0,
                       generator=torch.Generator().manual_seed(a.seed + 123456))
    if config['workers'] > 0:
        loader_args['prefetch_factor'] = 2
    loader = DataLoader(dataset, **loader_args)
    val_loader = DataLoader(EvalDataset(images,labels,val_ids),batch_size=config['eval_batch_size'],
                            num_workers=0,pin_memory=True,shuffle=False)
    model = WideResNet(config['depth'],config['width'])
    init_hash = __import__('hashlib').sha256(b''.join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(),lr=config['lr'],momentum=0.9,nesterov=True,weight_decay=5e-4)
    scaler = torch.amp.GradScaler('cuda',enabled=config['amp'])
    amp_health=AMPHealth(config['amp_monitor'])
    full_steps = len(loader)
    total_steps = config['epochs'] * full_steps
    start_epoch, best_nll, best_epoch, skips, carried_s = 0,float('inf'),-1,0,0.0
    last = run_dir/'last.pt'
    committed_history=[]
    best_model=None
    def write_best(checkpoint):
        best=dict(checkpoint)
        best['model']=checkpoint['best_model']
        best['epoch']=checkpoint['best_epoch']
        save_torch(run_dir/'best.pt',best)
    if last.exists() and not a.probe:
        ck = torch.load(last,map_location='cpu',weights_only=False)
        if ck['config_hash'] != cfg_hash or ck['method'] != a.method or ck['seed'] != a.seed:
            raise RuntimeError('Checkpoint/config mismatch')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start_epoch=ck['epoch']+1; best_nll=ck['best_nll']; best_epoch=ck['best_epoch']; skips=ck['amp_skips']
        amp_health.load_state_dict(ck['amp_health'])
        carried_s=ck['elapsed_s']
        committed_history=ck['committed_history'];best_model=ck['best_model']
        write_best(ck)
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
    torch.cuda.reset_peak_memory_stats()
    history = run_dir/'history.csv'
    columns=['epoch','time','epoch_s','train_s','val_s','elapsed_s','batches','samples','ce','jsd','beta',
             'total_loss','lr','val_accuracy','val_macro_f1','val_nll','amp_skips','amp_scale','amp_consecutive',
             'amp_window_skips','actual_updates','peak_allocated_mib','peak_reserved_mib']
    if start_epoch:
        # last.pt is the single atomic epoch commit, including full history and best weights.
        with history.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(committed_history)
    mode='a' if history.exists() and start_epoch else 'w'
    expected_epochs = min(config['epochs'],a.probe_epochs) if a.probe else config['epochs']
    timings=[]
    with history.open(mode,encoding='utf-8',newline='',buffering=1) as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        if mode=='w': writer.writeheader()
        for epoch in range(start_epoch,expected_epochs):
            safety_checkpoint()
            dataset.set_epoch(epoch)
            sampler_gen.manual_seed(a.seed * 100000 + epoch)
            model.train()
            torch.cuda.synchronize(); epoch_start=time.monotonic()
            totals=np.zeros(5,dtype=np.float64);batch_times=[];sample_count=0
            for batch,(x,y) in enumerate(loader):
                safety_checkpoint()
                step=epoch*full_steps+batch
                for group in optimizer.param_groups: group['lr']=config['lr']*lr_factor(step,total_steps)
                t=time.monotonic()
                y=y.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                beta=beta_value(a.method,step,total_steps)
                with torch.autocast('cuda',dtype=torch.float16,enabled=config['amp']):
                    if a.method in ('augmix','warmup','matched'):
                        x=torch.cat(x,dim=0).cuda(non_blocking=True)
                        output=model(x)
                        z=output.split(len(y),dim=0)
                        ce=F.cross_entropy(z[0].float(),y)
                        jsd=jsd_loss(z)
                        loss=ce+beta*jsd
                    else:
                        x=x.cuda(non_blocking=True)
                        jsd=torch.zeros((),device='cuda')
                        if a.method=='mixup':
                            rng=np.random.default_rng(np.random.SeedSequence([a.seed,epoch,batch,900]))
                            lam=float(rng.beta(1,1))
                            ix=torch.as_tensor(rng.permutation(len(y)),device='cuda')
                            output=model(lam*x+(1-lam)*x[ix])
                            ce=lam*F.cross_entropy(output.float(),y)+(1-lam)*F.cross_entropy(output.float(),y[ix])
                        else:
                            output=model(x);ce=F.cross_entropy(output.float(),y)
                        loss=ce
                if not torch.isfinite(loss).item():
                    raise RuntimeError('Non-finite loss at epoch %d batch %d' % (epoch,batch))
                old_scale=scaler.get_scale()
                scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                amp_health.update(scaler.get_scale()<old_scale,scaler.get_scale())
                skips=amp_health.total_skips
                # Synchronize benchmark steps only; the .item logging below synchronizes formal loss.
                totals += np.array([ce.item(),jsd.item(),beta,loss.item(),1])*len(y)
                sample_count += len(y)
                batch_times.append(time.monotonic()-t)
                if a.probe and a.max_batches and batch+1>=a.max_batches:
                    break
            torch.cuda.synchronize();train_s=time.monotonic()-epoch_start
            vstart=time.monotonic();metrics=validate(model,val_loader,config['amp'])
            torch.cuda.synchronize();val_s=time.monotonic()-vstart
            row={'epoch':epoch,'time':now(),'epoch_s':train_s+val_s,'train_s':train_s,'val_s':val_s,
                 'elapsed_s':carried_s+time.monotonic()-started,'batches':batch+1,'samples':sample_count,
                 'ce':totals[0]/sample_count,'jsd':totals[1]/sample_count,'beta':totals[2]/sample_count,
                 'total_loss':totals[3]/sample_count,'lr':optimizer.param_groups[0]['lr'],
                 'val_accuracy':metrics['accuracy'],'val_macro_f1':metrics['macro_f1'],'val_nll':metrics['nll'],
                 'amp_skips':skips,'amp_scale':amp_health.current_scale,'amp_consecutive':amp_health.consecutive,
                 'amp_window_skips':sum(amp_health.window),'actual_updates':amp_health.slots-amp_health.total_skips,
                 'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                 'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
            if not math.isfinite(metrics['nll']): raise RuntimeError('Non-finite validation NLL')
            improved=metrics['nll']<best_nll
            if improved:
                best_nll=metrics['nll'];best_epoch=epoch
                best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            committed_history.append(row)
            ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                'epoch':epoch,'best_nll':best_nll,'best_epoch':best_epoch,'config_hash':cfg_hash,
                'method':a.method,'seed':a.seed,'amp_skips':skips,'amp_health':amp_health.state_dict(),'elapsed_s':row['elapsed_s'],
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'init_hash':init_hash,
                'best_model':best_model,'committed_history':committed_history,
                'architecture':{'depth':config['depth'],'width':config['width']}}
            if not a.probe:
                save_torch(last,ck)
                if improved: write_best(ck)
            writer.writerow(row);f.flush();os.fsync(f.fileno())
            timings.append(row)
            write_json(run_dir/'progress.json',dict(row,best_epoch=best_epoch,status='running'))
            print(a.method,a.seed,'epoch',epoch+1,'/',config['epochs'],'val',round(metrics['accuracy'],4),
                  'nll',round(metrics['nll'],4),'seconds',round(train_s+val_s,1),flush=True)
    write_json(run_dir/('benchmark.json' if a.probe else 'completed.json'),
               {'finished':now(),'status':'probe_completed' if a.probe else 'completed','config_hash':cfg_hash,
                'elapsed_s':carried_s+time.monotonic()-started,'epochs':expected_epochs,'best_epoch':best_epoch,
                'best_val_nll':best_nll,'amp_skips':skips,'init_hash':init_hash,'timings':timings,
                'full_batches_per_epoch':full_steps,'parameters':sum(p.numel() for p in model.parameters()),
                'best_sha256':None if a.probe else digest(run_dir/'best.pt')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--method',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--probe',action='store_true')
    p.add_argument('--probe-epochs',type=int,default=1);p.add_argument('--max-batches',type=int,default=0)
    main(p.parse_args())
