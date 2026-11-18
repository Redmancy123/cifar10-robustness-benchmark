import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
METHODS = ['erm', 'mixup', 'augmix', 'warmup', 'matched']


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_fingerprints():
    return {str(p.relative_to(ROOT)).replace('\\','/'): digest(p)
            for folder in ('src','scripts') for p in sorted((ROOT/folder).glob('*.py'))}


def verify_frozen(config):
    if not config.get('frozen'):
        raise RuntimeError('Configuration is not frozen')
    if digest(ROOT/'data/data_manifest.json') != config['data_manifest_sha256']:
        raise RuntimeError('Data manifest changed since freeze')
    if source_fingerprints() != config['source_hashes']:
        raise RuntimeError('Source files changed since freeze; requires new review and freeze')
    for key in ('train','val'):
        if digest(ROOT/'data/splits'/(key+'_indices.npy')) != config['split_hashes'][key]:
            raise RuntimeError('Frozen split changed')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        # Allocator budget only, not a guarantee about all CUDA/WDDM allocations.
        torch.cuda.set_per_process_memory_fraction(0.5, 0)


def safety_checkpoint():
    """Supervisor owns only this run's STOP/PAUSE flags; never changes hardware."""
    import time
    control = os.environ.get('ROBUST_CONTROL_DIR')
    if not control:
        if torch.cuda.is_available():
            raise RuntimeError('GPU work requires the reviewed supervisor')
        return
    root = Path(control)
    (root / 'HEARTBEAT').touch()
    if (root / 'STOP').exists():
        raise RuntimeError('Supervisor requested stop: ' + (root / 'STOP').read_text(encoding='utf-8'))
    while (root / 'PAUSE').exists():
        (root / 'HEARTBEAT').touch()
        if (root / 'STOP').exists():
            raise RuntimeError('Supervisor requested stop while cooling')
        time.sleep(1)
