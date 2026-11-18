"""CIFAR preactivation WRN; architecture follows Wide Residual Networks (BMVC 2016)."""
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.shortcut = nn.Conv2d(cin, cout, 1, stride=stride, bias=False) if cin != cout or stride != 1 else None

    def forward(self, x):
        pre = torch.relu(self.bn1(x))
        shortcut = x if self.shortcut is None else self.shortcut(pre)
        out = self.conv1(pre)
        out = self.conv2(torch.relu(self.bn2(out)))
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth=16, width=2):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        widths = [16, 16 * width, 32 * width, 64 * width]
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        groups = []
        for stage in range(3):
            blocks = [Block(widths[stage], widths[stage+1], 1 if stage == 0 else 2)]
            blocks.extend(Block(widths[stage+1], widths[stage+1], 1) for _ in range(n-1))
            groups.append(nn.Sequential(*blocks))
        self.groups = nn.Sequential(*groups)
        self.bn = nn.BatchNorm2d(widths[-1])
        self.fc = nn.Linear(widths[-1], 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.groups(self.stem(x))
        x = torch.relu(self.bn(x)).mean(dim=(2, 3))
        return self.fc(x)
