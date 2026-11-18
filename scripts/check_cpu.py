"""Small CPU-only correctness checks; no dataset download or GPU allocation."""
import io
import json
import sys
import tarfile
import tempfile
import importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.common import ROOT,write_json,now
from src.model import WideResNet
from src.data import TrainDataset
from src.metrics import Metrics,jsd_loss,beta_value


def main():
    torch.set_num_threads(2)
    checks=[]
    torch.manual_seed(101)
    model=WideResNet();x=torch.randn(4,3,32,32);z=model(x)
    assert z.shape==(4,10)
    loss=torch.nn.functional.cross_entropy(z,torch.tensor([0,1,2,3]));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    checks.append('WRN shapes and finite CPU gradients')
    z=torch.randn(8,10,requires_grad=True)
    j=jsd_loss([z,z,z]);assert abs(j.item())<1e-6
    j2=jsd_loss([z,z+torch.randn_like(z),z+torch.randn_like(z)])
    assert j2.item()>=0;j2.backward();assert torch.isfinite(z.grad).all()
    checks.append('JSD identical-input zero, nonnegative and differentiable')
    extreme=[torch.tensor([[100.,-100.,-100.]],requires_grad=True),
             torch.tensor([[-100.,100.,-100.]],requires_grad=True),
             torch.tensor([[-100.,-100.,100.]],requires_grad=True)]
    extreme_loss=jsd_loss(extreme);extreme_loss.backward()
    assert torch.isfinite(extreme_loss) and all(torch.isfinite(z.grad).all() for z in extreme)
    checks.append('Extreme underflow probabilities retain finite JSD gradients')
    for total in [1,2,7,100,70000]:
        values=np.array([beta_value('warmup',t,total) for t in range(total)])
        assert np.isclose(values.mean(),beta_value('matched',0,total))
        assert (np.diff(values)>=0).all()
    checks.append('Exact discrete mean-weight match including tiny schedules')
    a=Metrics();b=Metrics()
    with torch.inference_mode():
        logits=torch.full((20,10),-5.0);labels=torch.arange(20)%10;logits[torch.arange(20),labels]=5.0
        a.add(logits,labels);b.add(logits[:7],labels[:7]);b.add(logits[7:],labels[7:])
    assert a.result()['accuracy']==1 and a.result()['macro_f1']==1
    assert abs(a.result()['nll']-b.result()['nll'])<1e-9
    assert abs(a.result()['ece']-b.result()['ece'])<1e-9
    checks.append('Global F1/NLL/ECE batch partition invariance')
    rng=np.random.default_rng(1);images=rng.integers(0,256,(4,32,32,3),dtype=np.uint8)
    d=TrainDataset(images,np.arange(4),np.arange(4),'warmup',17)
    first=d[0][0];again=d[0][0]
    assert all(torch.equal(a,b) for a,b in zip(first,again))
    d.set_epoch(1);changed=d[0][0]
    assert any(not torch.equal(a,b) for a,b in zip(first,changed))
    pair=TrainDataset(images,np.arange(4),np.arange(4),'augmix',17);pair.set_epoch(1)
    assert all(torch.equal(a,b) for a,b in zip(pair[0][0],changed))
    checks.append('Per-image/epoch/view determinism and AugMix/warmup paired inputs')
    # Validate shared epoch propagation through a real Windows persistent worker.
    loader=torch.utils.data.DataLoader(d,batch_size=2,num_workers=1,persistent_workers=True,shuffle=False)
    d.set_epoch(0);batch0=next(iter(loader))[0][0]
    d.set_epoch(1);batch1=next(iter(loader))[0][0]
    assert not torch.equal(batch0,batch1)
    del loader
    checks.append('Windows persistent worker sees shared epoch changes')
    spec=importlib.util.spec_from_file_location('prepare_data',ROOT/'scripts/prepare_data.py')
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    with tempfile.TemporaryDirectory(dir=ROOT/'runs') as folder:
        base=Path(folder);prep.RAW=base
        archive=base/'bad.tar'
        with tarfile.open(archive,'w') as tar:
            m=tarfile.TarInfo('../escape');m.size=1;tar.addfile(m,io.BytesIO(b'x'))
        try:prep.extract_checked(archive)
        except RuntimeError:pass
        else:raise AssertionError('Traversal accepted')
        archive=base/'good.tar';(base/'sample').write_bytes(b'BAD')
        with tarfile.open(archive,'w') as tar:
            m=tarfile.TarInfo('sample');m.size=3;tar.addfile(m,io.BytesIO(b'OK!'))
        prep.extract_checked(archive);assert (base/'sample').read_bytes()==b'OK!'
    checks.append('Archive traversal rejected and equal-size corruption replaced')
    write_json(ROOT/'runs/cpu_checks.json',{'time':now(),'status':'passed','checks':checks,
                                        'parameters':sum(p.numel() for p in model.parameters())})
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
