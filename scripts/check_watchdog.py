"""CPU-only mocked safety tests. Does not launch or heat a GPU."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import runtime
from src.common import ROOT,write_json,now


def sample(temperature):
    return {'temperature_c':temperature,'memory_used_mib':1000,'memory_total_mib':4096,'power_w':10}


def scenario(kind):
    with tempfile.TemporaryDirectory(dir=ROOT/'runs') as td:
        base=Path(td);calls=[];state={};samples=0;pause_states=[]
        class Child:
            pid=987654321
            def __init__(self,*args,**kwargs):
                self.code=None;self.control=Path(kwargs['env']['ROBUST_CONTROL_DIR']);state['child']=self
            def poll(self):return self.code
            def wait(self,timeout=None):
                calls.append(('wait',timeout))
                if self.code is None:raise subprocess.TimeoutExpired('mock-only',timeout)
                return self.code
        def terminate(child):
            calls.append(('terminate',child.pid));child.code=-9
        def telemetry():
            nonlocal samples
            samples+=1
            if samples==1:return sample(50)
            child=state['child']
            if kind=='thermal_ram_failure':return sample(85)
            if kind=='thermal':
                if samples==2:return sample(80)
                if samples==3:
                    pause_states.append((child.control/'PAUSE').exists());return sample(72)
                pause_states.append((child.control/'PAUSE').exists())
                (base/'runs/STOP_ALL').touch()
                return sample(85)
            if kind=='monitor_failure':raise RuntimeError('simulated unavailable sensor')
            if kind=='stop_all':(base/'runs/STOP_ALL').touch()
            if kind=='stale':
                pulse=child.control/'HEARTBEAT';pulse.touch();os.utime(pulse,(1,1))
            return sample(50)
        with patch.object(runtime,'ROOT',base),patch.object(runtime,'gpu_sample',telemetry),\
             patch.object(runtime,'free_ram_gib',side_effect=RuntimeError('RAM sensor failure') if kind=='thermal_ram_failure' else None,return_value=5.0),\
             patch.object(runtime.subprocess,'Popen',Child),patch.object(runtime,'terminate_owned',terminate),\
             patch.object(runtime.time,'sleep',return_value=None):
            try:runtime.run_guarded(['mock_never_executed.py'],kind,max_hours=1)
            except RuntimeError:pass
            else:raise AssertionError('Expected protective failure')
        status=json.loads(next(base.glob('runs/supervision/*/status.json')).read_text(encoding='utf-8'))
        assert status['status']=='failed' and any(c[0]=='terminate' for c in calls)
        if kind=='thermal':
            assert pause_states==[True,False]
            assert 'monitor_error' not in status
            assert not any(c==('wait',8) for c in calls),'85C incorrectly entered grace period'
        if kind=='thermal_ram_failure':
            assert samples==2 and not any(c==('wait',8) for c in calls)
            assert '85C' in status['reason']
        if kind=='monitor_failure':assert samples==4
        return kind+' passed'


if __name__=='__main__':
    checks=[scenario(k) for k in ['thermal','thermal_ram_failure','monitor_failure','stop_all','stale']]
    write_json(ROOT/'runs/watchdog_checks.json',{'status':'passed','time':now(),'checks':checks})
    print(json.dumps(checks,indent=2))
