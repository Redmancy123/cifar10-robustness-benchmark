"""Stateless per-image/epoch augmentation; no global NumPy RNG side effects.

AugMix operator semantics adapted from google-research/augmix (Apache-2.0).
See references/来源与引用.md; independently organized implementation below.
"""
import multiprocessing as mp
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from .common import ROOT, digest

BILINEAR = Image.Resampling.BILINEAR
AFFINE = Image.Transform.AFFINE


def verified_pickle(path):
    manifest = json.loads((ROOT / 'data/data_manifest.json').read_text(encoding='utf-8'))
    if digest(path) != manifest['cifar_files'][path.name]:
        raise RuntimeError('CIFAR file changed since archive validation: ' + str(path))
    with path.open('rb') as f:
        return pickle.load(f, encoding='bytes')


def load_train_pool():
    images, labels = [], []
    for k in range(1, 6):
        d = verified_pickle(ROOT / 'data/raw/cifar-10-batches-py' / ('data_batch_%d' % k))
        images.append(d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        labels.extend(d[b'labels'])
    return np.concatenate(images), np.array(labels, dtype=np.int64)


def load_test_pool():
    d = verified_pickle(ROOT / 'data/raw/cifar-10-batches-py/test_batch')
    return d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b'labels'], dtype=np.int64)


def to_tensor(a):
    a = np.asarray(a, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def base_augment(a, rng):
    padded = np.pad(a, ((4, 4), (4, 4), (0, 0)), mode='constant')
    i, j = rng.integers(0, 9, size=2)
    cropped = padded[i:i+32, j:j+32]
    if rng.random() < 0.5:
        cropped = cropped[:, ::-1]
    return Image.fromarray(np.ascontiguousarray(cropped))


def operation(image, op, rng, severity=3):
    if op == 0:
        return ImageOps.autocontrast(image)
    if op == 1:
        return ImageOps.equalize(image)
    level = rng.uniform(0.1, severity)
    if op == 2:
        return ImageOps.posterize(image, 4 - int(level * 4 / 10))
    if op == 3:
        angle = int(level * 30 / 10)
        angle *= -1 if rng.random() > 0.5 else 1
        return image.rotate(angle, resample=BILINEAR)
    if op == 4:
        return ImageOps.solarize(image, 256 - int(level * 256 / 10))
    if op in (5, 6):
        amount = level * 0.3 / 10
        amount *= -1 if rng.random() > 0.5 else 1
        affine = (1, amount, 0, 0, 1, 0) if op == 5 else (1, 0, 0, amount, 1, 0)
    elif op in (7, 8):
        amount = int(level * (32 / 3) / 10)
        amount *= -1 if rng.random() > 0.5 else 1
        affine = (1, 0, amount, 0, 1, 0) if op == 7 else (1, 0, 0, 0, 1, amount)
    else:
        raise ValueError(op)
    return image.transform((32, 32), AFFINE, affine, resample=BILINEAR)


def augmix(image, rng):
    weights = rng.dirichlet(np.ones(3)).astype(np.float32)
    mixture = np.zeros((32, 32, 3), dtype=np.float32)
    mixing = np.float32(rng.beta(1, 1))
    for weight in weights:
        branch = image.copy()
        for _ in range(int(rng.integers(1, 4))):
            branch = operation(branch, int(rng.integers(0, 9)), rng)
        mixture += weight * np.asarray(branch, dtype=np.float32)
    pixels = (1 - mixing) * np.asarray(image, dtype=np.float32) + mixing * mixture
    return to_tensor(pixels)


class TrainDataset(Dataset):
    def __init__(self, images, labels, indices, method, seed):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.method = method
        self.seed = seed
        # Shared memory propagates epoch changes to Windows persistent workers.
        self.epoch = mp.get_context('spawn').Value('i', 0)

    def set_epoch(self, epoch):
        self.epoch.value = epoch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        index = int(self.indices[j])
        epoch = self.epoch.value
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index, 0]))
        image = base_augment(self.images[index], rng)
        base = to_tensor(image)
        if self.method in ('augmix', 'warmup', 'matched'):
            views = [base]
            for view in (1, 2):
                vrng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index, view]))
                views.append(augmix(image, vrng))
            return tuple(views), int(self.labels[index])
        return base, int(self.labels[index])


class EvalDataset(Dataset):
    def __init__(self, images, labels, indices=None):
        self.images = images
        self.labels = labels
        self.indices = np.arange(len(labels)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        return to_tensor(self.images[i]), int(self.labels[i])
