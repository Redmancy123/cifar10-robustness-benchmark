import numpy as np
import torch
from torch.nn import functional as F


def jsd_loss(logits):
    # Avoid kl_div's derivative of a probability target that may underflow to zero.
    log_probabilities = [F.log_softmax(z.float(), dim=1) for z in logits]
    log_mean = torch.logsumexp(torch.stack(log_probabilities),dim=0) - np.log(3.0)
    return sum((lp.exp() * (lp-log_mean)).sum(dim=1).mean() for lp in log_probabilities) / 3


def beta_value(method, step, total):
    warm = max(1, int(np.ceil(0.2 * total)))
    if method == 'warmup':
        return 12.0 * min(step / warm, 1.0)
    if method == 'matched':
        # Exact discrete average of beta(t), t=0,...,T-1 (warm <= T).
        return 12.0 * (1 - (warm + 1) / (2 * total))
    return 12.0 if method == 'augmix' else 0.0


def lr_factor(step, total):
    warm = max(1, int(np.ceil(total * 0.05)))
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, total - 1 - warm)
    return 1e-3 + (1 - 1e-3) * (1 + np.cos(np.pi * progress)) / 2


class Metrics:
    def __init__(self):
        self.confusion = np.zeros((10, 10), dtype=np.int64)
        self.n = 0
        self.loss_sum = 0.0
        self.bin_n = np.zeros(15, dtype=np.int64)
        self.bin_conf = np.zeros(15, dtype=np.float64)
        self.bin_correct = np.zeros(15, dtype=np.int64)

    def add(self, logits, targets):
        logp = F.log_softmax(logits.float(), dim=1)
        confidence, predicted = logp.exp().max(1)
        loss = F.nll_loss(logp, targets, reduction='none')
        y = targets.cpu().numpy()
        pred = predicted.cpu().numpy()
        conf = confidence.cpu().numpy()
        correct = pred == y
        self.confusion += np.bincount(y * 10 + pred, minlength=100).reshape(10, 10)
        self.n += len(y)
        self.loss_sum += loss.double().sum().item()
        bins = np.clip(np.ceil(conf * 15).astype(int) - 1, 0, 14)
        self.bin_n += np.bincount(bins, minlength=15)
        self.bin_conf += np.bincount(bins, weights=conf, minlength=15)
        self.bin_correct += np.bincount(bins, weights=correct, minlength=15).astype(np.int64)
        return pred, conf, loss.cpu().numpy()

    def result(self):
        assert self.n > 0
        tp = self.confusion.diagonal()
        den = self.confusion.sum(0) + self.confusion.sum(1)
        f1 = np.divide(2 * tp, den, out=np.zeros(10), where=den > 0)
        return {'n': self.n, 'correct': int(tp.sum()), 'accuracy': float(tp.sum()/self.n),
                'macro_f1': float(f1.mean()), 'nll': self.loss_sum/self.n,
                'ece': float(np.abs(self.bin_correct-self.bin_conf).sum()/self.n),
                'confusion': self.confusion.tolist(), 'bin_n': self.bin_n.tolist(),
                'bin_conf_sum': self.bin_conf.tolist(), 'bin_correct': self.bin_correct.tolist()}
