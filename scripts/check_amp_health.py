"""CPU checks for the versioned AMP monitoring amendment."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.amp_health import AMPHealth
from src.common import ROOT,now,write_json

POLICY={'window':2000,'max_window_skip_fraction':0.01,'max_consecutive_skips':8,'minimum_scale':1.0}


def must_fail(call):
    try:call()
    except RuntimeError:return
    raise AssertionError('Health failure was not detected')


def main():
    checks=[];m=AMPHealth(POLICY)
    for i in range(70000):m.update(i%2500==0,65536.)
    assert m.total_skips>20
    checks.append('Sparse normal recalibration exceeding 20 total skips remains valid')
    m=AMPHealth(POLICY)
    for i in range(7):m.update(True,65536.)
    must_fail(lambda:m.update(True,32768.));checks.append('Eight consecutive skips trigger')
    m=AMPHealth(POLICY)
    for i in range(2000):m.update(i%100==0,65536.)
    assert sum(m.window)==20
    m=AMPHealth(POLICY)
    for i in range(1999):m.update(i%99==0,65536.)
    must_fail(lambda:m.update(False,65536.));checks.append('Window boundary allows 20/2000 and rejects 21/2000')
    for value in [0.5,float('nan'),float('inf')]:
        must_fail(lambda value=value:AMPHealth(POLICY).update(False,value))
    checks.append('Scale below one, NaN and Inf trigger')
    a=AMPHealth(POLICY)
    for i in range(2100):a.update(i%300==0,65536.)
    b=AMPHealth(POLICY);b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    for i in range(100):a.update(i%300==0,65536.);b.update(i%300==0,65536.)
    assert a.state_dict()==b.state_dict();checks.append('Checkpoint state retains window, counters and scale')
    p=torch.nn.Parameter(torch.tensor(1.));optimizer=torch.optim.SGD([p],lr=.1)
    scaler=torch.amp.GradScaler('cpu');scaler.scale(p.square()).backward();p.grad.fill_(float('inf'))
    old=p.detach().clone();scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
    assert torch.equal(p,old) and scaler.get_scale()<scale
    checks.append('CPU GradScaler inf injection skips parameter update and backs off')
    write_json(ROOT/'runs/amp_health_checks.json',{'time':now(),'status':'passed','checks':checks})
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
