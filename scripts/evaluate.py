"""Final evaluation only; never changes a checkpoint or any training configuration."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.common import ROOT,CORRUPTIONS,configure_torch,digest,now,safety_checkpoint,write_json,verify_frozen
from src.data import EvalDataset,load_test_pool
from src.model import WideResNet
from src.metrics import Metrics


@torch.inference_mode()
def infer(model,images,labels,batch_size,amp):
    loader=DataLoader(EvalDataset(images,labels),batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)
    metrics=Metrics();pred=[];conf=[];nll=[]
    for x,y in loader:
        safety_checkpoint()
        x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp): z=model(x)
        p,c,l=metrics.add(z,y);pred.append(p);conf.append(c);nll.append(l)
    return metrics.result(),{'pred':np.concatenate(pred),'confidence':np.concatenate(conf),'nll':np.concatenate(nll)}


def main(a):
    configure_torch();safety_checkpoint()
    run=Path(a.run).resolve()
    if not run.is_relative_to((ROOT/'runs/formal').resolve()): raise RuntimeError('Only project formal runs may be evaluated')
    cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
    verify_frozen(cfg)
    if not (run/'completed.json').exists():raise RuntimeError('Incomplete training run')
    source=run/'best.pt';checkpoint_hash=digest(source)
    completed=json.loads((run/'completed.json').read_text(encoding='utf-8'))
    if completed['best_sha256']!=checkpoint_hash:
        raise RuntimeError('Checkpoint differs from training completion record')
    ck=torch.load(source,map_location='cpu',weights_only=False)
    if (ck['config_hash']!=cfg['config_hash'] or ck['epoch']!=ck['best_epoch']
            or ck['method']!=cfg['method'] or ck['seed']!=cfg['seed']):
        raise RuntimeError('Selected checkpoint does not match validation-best identity')
    result_dir=ROOT/'results/predictions'/run.name;result_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/data_manifest.json').read_text(encoding='utf-8'))
    if digest(ROOT/'data/data_manifest.json')!=cfg['data_manifest_sha256']:
        raise RuntimeError('Data manifest changed after configuration freeze')
    model=WideResNet(cfg['depth'],cfg['width']);model.load_state_dict(ck['model']);model=model.cuda().eval()
    images,labels=load_test_pool()
    rows=[];started=time.monotonic()
    def condition(name,severity,x,y,ids):
        safety_checkpoint()
        token='%s_%d'%(name,severity)
        metric_path=result_dir/(token+'.json');pred_path=result_dir/(token+'.npz')
        if metric_path.exists() and pred_path.exists():
            existing=json.loads(metric_path.read_text(encoding='utf-8'))
            if existing['checkpoint_sha256']!=checkpoint_hash:raise RuntimeError('Stale cached evaluation')
            rows.append(existing);return
        metrics,prediction=infer(model,x,y,cfg['eval_batch_size'],cfg['amp'])
        np.savez_compressed(pred_path,base_id=ids,y=y,**prediction)
        row=dict(metrics,method=cfg['method'],seed=cfg['seed'],run_id=run.name,corruption=name,
                 severity=severity,checkpoint_sha256=checkpoint_hash,selected_epoch=ck['epoch'],
                 evaluation_profile=cfg['evaluation_profile'])
        write_json(metric_path,row);rows.append(row)
        print(run.name,name,severity,round(metrics['accuracy'],4),flush=True)
    condition('clean',0,images,labels,np.arange(10000))
    if cfg['evaluation_profile']=='full':
        ids=np.arange(10000)
    else:
        ids=np.load(ROOT/'data/splits/corruption_subset.npy')
        if digest(ROOT/'data/splits/corruption_subset.npy')!=cfg['corruption_subset_sha256']:
            raise RuntimeError('Subset mismatch')
        condition('clean_subset',0,images[ids],labels[ids],ids)
    cdir=ROOT/'data/raw/CIFAR-10-C'
    if digest(cdir/'labels.npy')!=manifest['test_labels_sha256']:raise RuntimeError('Corruption labels changed')
    for c in CORRUPTIONS:
        safety_checkpoint()
        path=cdir/(c+'.npy')
        if digest(path)!=manifest['corruptions'][c]['sha256']:raise RuntimeError('Corruption file changed: '+c)
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        for severity in range(1,6):
            block=array[(severity-1)*10000:severity*10000]
            x=block if len(ids)==10000 else block[ids]
            condition(c,severity,x,labels[ids],ids)
        del array
    write_json(result_dir/'evaluation_complete.json',{'finished':now(),'elapsed_s':time.monotonic()-started,
               'conditions':len(rows),'checkpoint_sha256':checkpoint_hash,'run_id':run.name})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args())
