"""Rebuild tables and scientific plots from all frozen runs, preserving negative results."""
import csv
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import ROOT,CORRUPTIONS,canonical_hash,write_json,now,verify_frozen


def table(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(config_path):
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'));verify_frozen(cfg)
    methods=cfg['methods'];seeds=cfg['seeds'];h=canonical_hash(cfg)[:10]
    out=ROOT/'results/tables';out.mkdir(parents=True,exist_ok=True)
    figs=ROOT/'results/figures';figs.mkdir(parents=True,exist_ok=True)
    rows=[];per_seed=[];histories={};all_conditions={}
    for m in methods:
        for seed in seeds:
            name='%s_seed%d_%s'%(m,seed,h);run=ROOT/'runs/formal'/name;pdir=ROOT/'results/predictions'/name
            complete=json.loads((pdir/'evaluation_complete.json').read_text(encoding='utf-8'))
            conditions=[json.loads(p.read_text(encoding='utf-8')) for p in pdir.glob('*.json') if p.name!='evaluation_complete.json']
            if len(conditions)!=76 or complete['conditions']!=76:raise RuntimeError('Missing final conditions')
            expected={('clean',0)}|{(c,s) for c in CORRUPTIONS for s in range(1,6)}
            if {(r['corruption'],r['severity']) for r in conditions}!=expected:
                raise RuntimeError('Wrong or duplicated condition keys')
            if any(r['n']!=10000 or r['checkpoint_sha256']!=complete['checkpoint_sha256'] for r in conditions):
                raise RuntimeError('Wrong sample count or checkpoint identity in final conditions')
            all_conditions[m,seed]=conditions
            with (run/'history.csv').open(encoding='utf-8',newline='') as f:history=list(csv.DictReader(f))
            histories[m,seed]=history
            clean=next(r for r in conditions if r['corruption']=='clean')
            corrupt=[r for r in conditions if r['corruption'] in CORRUPTIONS]
            if len(corrupt)!=75:raise RuntimeError('Expected 75 corruption conditions')
            supervision=[]
            for p in (ROOT/'runs/supervision').glob('*/status.json'):
                s=json.loads(p.read_text(encoding='utf-8'))
                if s['tag']=='train_'+name:
                    if 'elapsed_s' not in s:raise RuntimeError('Training supervisor is not finalized')
                    supervision.append(s)
            if not supervision:raise RuntimeError('Missing actual training wall-clock evidence')
            row={'method':m,'seed':seed,'clean_accuracy':clean['accuracy'],'clean_macro_f1':clean['macro_f1'],
                 'corruption_accuracy':float(np.mean([r['accuracy'] for r in corrupt])),
                 'corruption_macro_f1':float(np.mean([r['macro_f1'] for r in corrupt])),
                 'corruption_nll':float(np.mean([r['nll'] for r in corrupt])),
                 'corruption_ece':float(np.mean([r['ece'] for r in corrupt])),
                 'training_wall_s_all_attempts':sum(s['elapsed_s'] for s in supervision),
                 'successful_internal_s':float(history[-1]['elapsed_s']),
                 'selected_epoch':clean['selected_epoch'],'evaluation_profile':cfg['evaluation_profile']}
            per_seed.append(row)
            for r in conditions:
                rows.append({k:r[k] for k in ('method','seed','run_id','corruption','severity','n','accuracy','macro_f1','nll','ece','checkpoint_sha256')})
    table(out/'metrics_by_condition.csv',rows);table(out/'metrics_by_seed.csv',per_seed)
    summary=[]
    keys=['clean_accuracy','clean_macro_f1','corruption_accuracy','corruption_macro_f1','corruption_nll','corruption_ece','training_wall_s_all_attempts']
    for m in methods:
        r={'method':m,'n_seeds':len(seeds),'epochs':cfg['epochs'],'evaluation_profile':cfg['evaluation_profile']}
        for k in keys:
            v=[x[k] for x in per_seed if x['method']==m];r[k+'_mean']=float(np.mean(v));r[k+'_std']=float(np.std(v,ddof=1))
        summary.append(r)
    table(out/'summary.csv',summary)
    pairs=[('augmix','erm'),('warmup','augmix'),('warmup','matched')];differences=[]
    for a,b in pairs:
        vals=[]
        for seed in seeds:
            av=next(r['corruption_accuracy'] for r in per_seed if r['method']==a and r['seed']==seed)
            bv=next(r['corruption_accuracy'] for r in per_seed if r['method']==b and r['seed']==seed)
            vals.append(av-bv);differences.append({'comparison':a+' - '+b,'seed':seed,'delta_pp':100*(av-bv)})
        differences.append({'comparison':a+' - '+b,'seed':'mean +/- sample SD',
                            'delta_pp':'%.6f +/- %.6f'%(100*np.mean(vals),100*np.std(vals,ddof=1))})
    table(out/'paired_differences.csv',differences)
    plt.rcParams.update({'figure.dpi':140,'savefig.dpi':300,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7']
    def save(fig,name):
        fig.tight_layout();fig.savefig(figs/(name+'.png'),bbox_inches='tight');fig.savefig(figs/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for m,color in zip(methods,colors):
        for ax,key in zip(axes,['accuracy','macro_f1']):
            values=np.array([[np.mean([r[key] for r in all_conditions[m,seed] if r['severity']==s and r['corruption'] in CORRUPTIONS]) for s in range(1,6)] for seed in seeds])*100
            ax.errorbar(range(1,6),values.mean(0),yerr=values.std(0,ddof=1),label=m,color=color,marker='o',capsize=3)
            ax.set(xlabel='Corruption severity level',ylabel=key+' (%)',xticks=range(1,6));ax.grid(alpha=.18)
    axes[0].legend();fig.suptitle('Mean over 15 types; error bars: sample SD across 3 training seeds');save(fig,'F03_severity_curves')
    heat=np.array([[np.mean([r['accuracy'] for seed in seeds for r in all_conditions[m,seed] if r['corruption']==c])*100 for c in CORRUPTIONS] for m in methods])
    fig,ax=plt.subplots(figsize=(13,4));im=ax.imshow(heat,vmin=0,vmax=100,cmap='viridis',aspect='auto')
    ax.set(yticks=range(len(methods)),yticklabels=methods,xticks=range(15),xticklabels=CORRUPTIONS)
    plt.setp(ax.get_xticklabels(),rotation=50,ha='right');fig.colorbar(im,ax=ax,label='Accuracy (%)');save(fig,'F04_corruption_heatmap')
    fig,ax=plt.subplots(figsize=(7,4))
    for i,(a,b) in enumerate(pairs):
        vals=[r['delta_pp'] for r in differences if r['comparison']==a+' - '+b and isinstance(r['seed'],int)]
        ax.scatter(np.full(3,i)+np.array([-.08,0,.08]),vals,color=colors[i]);ax.errorbar(i,np.mean(vals),yerr=np.std(vals,ddof=1),color='black',capsize=5,marker='D')
    ax.axhline(0,color='gray',lw=1);ax.set(xticks=range(3),xticklabels=[a+' - '+b for a,b in pairs],ylabel='Paired CAcc difference (percentage points)');save(fig,'F05_paired_differences')
    for name,xkey,ykey,xlabel,ylabel in [('F06_clean_vs_corruption','clean_accuracy','corruption_accuracy','Clean accuracy (%)','Corruption accuracy (%)'),
                                        ('F07_quality_vs_cost','training_wall_s_all_attempts','corruption_accuracy','Actual training wall time (hours)','Corruption accuracy (%)')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for m,color in zip(methods,colors):
            data=[r for r in per_seed if r['method']==m];x=[r[xkey]*(100 if xkey=='clean_accuracy' else 1/3600) for r in data];y=[r[ykey]*100 for r in data]
            ax.scatter(x,y,label=m,color=color,s=45)
        ax.set(xlabel=xlabel,ylabel=ylabel);ax.legend();ax.grid(alpha=.18);save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for m,color in zip(methods,colors):
        for ax,key in zip(axes,['val_accuracy','val_nll']):
            values=np.array([[float(r[key]) for r in histories[m,s]] for s in seeds]);mean=values.mean(0);sd=values.std(0,ddof=1);x=np.arange(1,len(mean)+1)
            ax.plot(x,mean,label=m,color=color);ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.12);ax.set(xlabel='Epoch',ylabel=key)
    axes[0].legend();save(fig,'F08_validation_learning_curves')
    # Fixed illustrative image ID and three predeclared types; not selected for favorable predictions.
    from src.data import load_test_pool
    images,labels=load_test_pool();image_id=1234;types=['gaussian_noise','motion_blur','jpeg_compression']
    fig,axes=plt.subplots(3,6,figsize=(10,5.5))
    for i,c in enumerate(types):
        array=np.load(ROOT/'data/raw/CIFAR-10-C'/(c+'.npy'),mmap_mode='r')
        for j in range(6):
            axes[i,j].imshow(images[image_id] if j==0 else array[(j-1)*10000+image_id],interpolation='nearest');axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title('Clean' if j==0 else 'Level '+str(j))
            if j==0:axes[i,j].set_ylabel(c)
    save(fig,'F02_corruption_examples')
    names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    categories={'ERM correct / Warmup wrong':[],'ERM wrong / Warmup correct':[],'Both wrong':[]}
    dirs={m:ROOT/'results/predictions'/('%s_seed17_%s'%(m,h)) for m in ('erm','warmup')}
    for c in types:
        for severity in (3,5):
            a=np.load(dirs['erm']/('%s_%d.npz'%(c,severity)))
            b=np.load(dirs['warmup']/('%s_%d.npz'%(c,severity)))
            if not np.array_equal(a['base_id'],b['base_id']):raise RuntimeError('Paired prediction IDs differ')
            ca=a['pred']==a['y'];cb=b['pred']==b['y']
            masks=[ca & ~cb,~ca & cb,~ca & ~cb]
            for (category,cases),mask in zip(categories.items(),masks):
                for j in np.flatnonzero(mask):
                    if len(cases)>=3:break
                    cases.append((c,severity,int(a['base_id'][j]),int(a['y'][j]),int(a['pred'][j]),int(b['pred'][j])))
    fig,axes=plt.subplots(3,3,figsize=(10,8))
    for i,(category,cases) in enumerate(categories.items()):
        for j,ax in enumerate(axes[i]):
            ax.set_xticks([]);ax.set_yticks([])
            if j>=len(cases):ax.text(.5,.5,'No candidate',ha='center');continue
            c,s,base,y,pa,pb=cases[j];array=np.load(ROOT/'data/raw/CIFAR-10-C'/(c+'.npy'),mmap_mode='r')
            ax.imshow(array[(s-1)*10000+base],interpolation='nearest')
            ax.set_title('%s L%d ID%d\ntrue: %s | ERM: %s\nWarmup: %s'%(c,s,base,names[y],names[pa],names[pb]),fontsize=8)
            if j==0:ax.set_ylabel(category)
    save(fig,'F09_failure_and_improvement_cases')
    write_json(ROOT/'results/completion.json',{'status':'completed','time':now(),'runs':len(per_seed),'n_seeds':3,
               'epochs':cfg['epochs'],'n_conditions':len(rows),'figures':sorted(p.name for p in figs.glob('*.png')),
               'interpretation':'No significance or originality claim; report positive, null and negative outcomes.'})
    print('All tables and figures rebuilt from completed runs.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/frozen.json')
    main(p.parse_args().config)
