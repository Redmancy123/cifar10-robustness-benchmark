"""External GPU/heartbeat watchdog. Never adjusts clocks, driver or power settings."""
import ctypes
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .common import ROOT, now, write_json


def gpu_sample():
    r = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,memory.used,memory.total,power.draw',
                        '--format=csv,noheader,nounits', '--id=0'], capture_output=True, text=True, timeout=8)
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    parts = r.stdout.strip().split(',')
    return {'temperature_c': float(parts[0]), 'memory_used_mib': float(parts[1]),
            'memory_total_mib': float(parts[2]), 'power_w': None if 'N/A' in parts[3] else float(parts[3])}


def free_ram_gib():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                    ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                    ('avail_extended', ctypes.c_ulonglong)]
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise RuntimeError('Cannot read available RAM')
    return value.avail_phys / 2**30


def terminate_owned(child):
    if child.poll() is None:
        # PID comes exclusively from our Popen, never from a process-name search.
        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                       capture_output=True, timeout=20)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def run_guarded(arguments, tag, max_hours=24):
    folder = ROOT / 'runs/supervision' / (tag + '_' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    initial = gpu_sample()
    if initial['temperature_c'] >= 80 or initial['memory_total_mib'] - initial['memory_used_mib'] < 700:
        raise RuntimeError('Insufficient cool/free GPU headroom before launch: ' + str(initial))
    env = os.environ.copy()
    env.update({'ROBUST_CONTROL_DIR': str(folder), 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8',
                'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    status = {'tag': tag, 'started': now(), 'command': [sys.executable] + list(arguments),
              'status': 'running', 'initial_gpu': initial, 'thresholds_c': {'pause': 80, 'resume': 72, 'stop': 85}}
    started = time.monotonic()
    child = None
    failures = 0
    paused_since = None
    reason = None
    write_json(folder / 'status.json', status)
    try:
        with (folder / 'stdout.log').open('w', encoding='utf-8', buffering=1) as out:
            child = subprocess.Popen([sys.executable] + list(arguments), cwd=ROOT, env=env,
                                     stdout=out, stderr=subprocess.STDOUT)
            status['pid'] = child.pid
            write_json(folder / 'status.json', status)
            with (folder / 'telemetry.jsonl').open('a', encoding='utf-8', buffering=1) as log:
                import json
                while child.poll() is None:
                    hard_stop = False
                    try:
                        sample = gpu_sample()
                        # Latch a known thermal hazard before any other sensor can fail.
                        if sample['temperature_c'] >= 85:
                            hard_stop = True
                            reason = 'GPU temperature reached conservative stop threshold 85C'
                        sample.update({'time': now(), 'ram_free_gib': free_ram_gib()})
                        failures = 0
                        log.write(json.dumps(sample) + '\n')
                        status['latest'] = sample
                        if not hard_stop and (sample['temperature_c'] >= 80 or sample['ram_free_gib'] < 1.0):
                            (folder / 'PAUSE').write_text('cooling or memory pressure', encoding='utf-8')
                            paused_since = paused_since or time.monotonic()
                        elif not hard_stop and paused_since is not None and sample['temperature_c'] <= 72 and sample['ram_free_gib'] >= 2.0:
                            (folder / 'PAUSE').unlink(missing_ok=True)
                            paused_since = None
                        if not hard_stop and paused_since is not None and time.monotonic() - paused_since > 900:
                            reason = 'Cooling/memory recovery exceeded 15 minutes'
                    except Exception as e:
                        failures += 1
                        status['monitor_error'] = repr(e)
                        if failures >= 3 and not hard_stop:
                            reason = 'Three consecutive GPU/RAM telemetry failures'
                    pulse = folder / 'HEARTBEAT'
                    age = time.time() - (pulse.stat().st_mtime if pulse.exists() else time.time() - (time.monotonic()-started))
                    if not hard_stop and paused_since is None and age > 240:
                        reason = 'Child heartbeat stale for 240 seconds'
                    if not hard_stop and (ROOT / 'runs/STOP_ALL').exists():
                        reason = 'User/project STOP_ALL requested'
                    if not hard_stop and time.monotonic() - started > max_hours * 3600:
                        reason = 'Job exceeded predeclared maximum runtime'
                    if reason:
                        (folder / 'STOP').write_text(reason, encoding='utf-8')
                        if hard_stop:
                            terminate_owned(child)
                            break
                        # Grace period lets a responsive worker flush records; then terminate only our tree.
                        try:
                            child.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            terminate_owned(child)
                        break
                    status['cooling'] = paused_since is not None
                    write_json(folder / 'status.json', status)
                    time.sleep(3)
            code = child.wait()
        status.update({'finished': now(), 'returncode': code,
                       'elapsed_s': time.monotonic()-started,
                       'status': 'completed' if code == 0 and not reason else 'failed', 'reason': reason})
        write_json(folder / 'status.json', status)
        if code or reason:
            raise RuntimeError('Guarded job failed; inspect %s: %s' % (folder, reason or code))
        return status
    finally:
        if child is not None and child.poll() is None:
            terminate_owned(child)
