"""Download public CIFAR archives; verify before manually extracting regular files."""
import hashlib
import json
import os
import pickle
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, CORRUPTIONS, digest, now, write_json
import numpy as np

RAW = ROOT / 'data/raw'
DOWNLOADS = [
    ('cifar-10-python.tar.gz', 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
     'c58f30108f718f92721af3b95e74349a'),
    ('CIFAR-10-C.tar', 'https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1',
     '56bf5dcef84df0e2308c6dcbcbbd8499')]


def md5(path):
    h = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(name, url, expected):
    path = RAW / name
    if path.exists():
        if md5(path) != expected:
            raise RuntimeError('Existing archive failed MD5: ' + str(path))
        return path
    partial = path.with_suffix(path.suffix + '.partial')
    if partial.exists() and md5(partial) == expected:
        partial.replace(path)
        return path
    for attempt in range(5):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'CIFAR-coursework/1.0'}
            if start:
                headers['Range'] = 'bytes=%d-' % start
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                ranged = response.status == 206
                if ranged and not response.headers.get('Content-Range', '').startswith('bytes %d-' % start):
                    raise RuntimeError('Unexpected Content-Range')
                if not ranged:
                    start = 0
                with partial.open('ab' if ranged and start else 'wb') as f:
                    last = time.monotonic()
                    count = start
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        count += len(chunk)
                        if time.monotonic() - last > 10:
                            print(name, 'downloaded MiB', round(count / 2**20), flush=True)
                            last = time.monotonic()
            if md5(partial) != expected:
                raise RuntimeError('Downloaded archive failed MD5; keep .partial for investigation')
            partial.replace(path)
            return path
        except Exception as e:
            print('Download attempt', attempt + 1, repr(e), flush=True)
            if attempt == 4 or isinstance(e, RuntimeError):
                raise
            time.sleep(min(2 ** attempt, 10))


def extract_checked(path):
    base = RAW.resolve()
    with tarfile.open(path, 'r:*') as tar:
        members = tar.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or '\\' in member.name or ':' in member.name:
                raise RuntimeError('Unsafe archive path: ' + member.name)
            target = (base / Path(*name.parts)).resolve()
            if not target.is_relative_to(base) or not (member.isdir() or member.isfile()):
                raise RuntimeError('Archive links/special files forbidden: ' + member.name)
        for member in members:
            target = base / Path(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + '.extracting')
                with tar.extractfile(member) as source, tmp.open('wb') as dest:
                    shutil.copyfileobj(source, dest, length=4 * 1024 * 1024)
                if tmp.stat().st_size != member.size:
                    raise RuntimeError('Incomplete archive extraction: ' + member.name)
                os.replace(tmp, target)


def load_cifar():
    folder = RAW / 'cifar-10-batches-py'
    batches = []
    labels = []
    for i in range(1, 6):
        # Only deserialize the public archive after its published MD5 was checked.
        with (folder / ('data_batch_%d' % i)).open('rb') as f:
            d = pickle.load(f, encoding='bytes')
        batches.append(d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        labels.extend(d[b'labels'])
    with (folder / 'test_batch').open('rb') as f:
        d = pickle.load(f, encoding='bytes')
    return np.concatenate(batches), np.array(labels, dtype=np.int64), d


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(RAW).free < 15 * 2**30:
        raise RuntimeError('Need at least 15 GiB free before preparation')
    for name, url, expected in DOWNLOADS:
        print('Preparing', name, flush=True)
        extract_checked(download(name, url, expected))
    images, labels, test = load_cifar()
    if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
        raise RuntimeError('Unexpected CIFAR training array')
    if not np.array_equal(np.bincount(labels), np.full(10, 5000)):
        raise RuntimeError('Unexpected training class counts')
    if test[b'data'].shape != (10000, 3072) or test[b'data'].dtype != np.uint8:
        raise RuntimeError('Unexpected CIFAR test array')
    if not np.array_equal(np.bincount(test[b'labels']), np.full(10, 1000)):
        raise RuntimeError('Unexpected test class counts')
    indices_dir = ROOT / 'data/splits'
    indices_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261117)
    train, val = [], []
    for k in range(10):
        ids = rng.permutation(np.flatnonzero(labels == k))
        train.extend(ids[:4500]); val.extend(ids[4500:])
    train, val = np.sort(train), np.sort(val)
    if len(set(train) & set(val)) or len(train) + len(val) != 50000:
        raise RuntimeError('Invalid split overlap or coverage')
    for name, ids in [('train_indices', train), ('val_indices', val)]:
        p = indices_dir / (name + '.npy')
        if p.exists() and not np.array_equal(np.load(p), ids):
            raise RuntimeError('Refusing to replace different frozen indices')
        np.save(p, ids)
    ytest = np.array(test[b'labels'])
    cdir = RAW / 'CIFAR-10-C'
    cy = np.load(cdir / 'labels.npy', allow_pickle=False)
    if cy.shape != (50000,) or not np.array_equal(cy, np.tile(ytest, 5)):
        raise RuntimeError('Corruption labels do not match the five original test label blocks')
    corruption_files = {}
    for c in CORRUPTIONS:
        p = cdir / (c + '.npy')
        a = np.load(p, mmap_mode='r', allow_pickle=False)
        if a.shape != (50000, 32, 32, 3) or a.dtype != np.uint8:
            raise RuntimeError('Unexpected corruption array: ' + c)
        corruption_files[c] = {'shape': list(a.shape), 'sha256': digest(p)}
    manifest = {'created': now(), 'split_seed': 20261117, 'train_n': 45000, 'val_n': 5000,
                'test_n': 10000, 'train_indices_sha256': digest(indices_dir / 'train_indices.npy'),
                'val_indices_sha256': digest(indices_dir / 'val_indices.npy'),
                'archives': {n: {'url': u, 'md5': h, 'sha256': digest(RAW / n)} for n, u, h in DOWNLOADS},
                'corruptions': corruption_files, 'test_labels_sha256': digest(cdir / 'labels.npy'),
                'cifar_files': {p.name: digest(p) for p in (RAW / 'cifar-10-batches-py').iterdir() if p.is_file()}}
    write_json(ROOT / 'data/data_manifest.json', manifest)
    print('Data prepared and verified; NO model predictions evaluated.', flush=True)


if __name__ == '__main__':
    main()
