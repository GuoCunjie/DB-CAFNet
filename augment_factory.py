# augment_factory.py
import torch
import torch.nn as nn
import torchaudio
import random

class MultiMask(nn.Module):
    def __init__(self, freq_mask=10, time_mask=20, num=3):
        super().__init__()
        self.fs = [torchaudio.transforms.FrequencyMasking(freq_mask) for _ in range(num)]
        self.ts = [torchaudio.transforms.TimeMasking(time_mask) for _ in range(num)]

    def forward(self, x):
        for f, t in zip(self.fs, self.ts):
            x = f(t(x))
        return x

class MelWarp(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.width = width

    def forward(self, x):
        if x.ndim == 3:  # [B, 128, T]
            x = x.unsqueeze(1)
        elif x.ndim == 5:  # 错误多加了一维
            x = x.squeeze(1)
        B, C, H, W = x.shape
        for b in range(B):
            start = random.randint(0, W - self.width)
            x[b, :, :, start:start + self.width] *= 0.7
        return x


class FrameDrop(nn.Module):
    def __init__(self, drop_prob=0.2):
        super().__init__()
        self.prob = drop_prob

    def forward(self, x):  
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 5:
            x = x.squeeze(1)
        B, C, H, W = x.shape
        for b in range(B):
            if random.random() < self.prob:
                drop_len = int(H * 0.1)
                start = random.randint(0, H - drop_len)
                x[b, :, start:start + drop_len, :] = 0
        return x



def get_augment_module(name):
    if name == "none":
        return nn.Identity()
    elif name == "specaugment":
        return nn.Sequential(
            torchaudio.transforms.FrequencyMasking(10),
            torchaudio.transforms.TimeMasking(20)
        )
    elif name == "strong_freq":
        return nn.Sequential(
            torchaudio.transforms.FrequencyMasking(40),
            torchaudio.transforms.TimeMasking(20)
        )
    elif name == "strong_time":
        return nn.Sequential(
            torchaudio.transforms.FrequencyMasking(10),
            torchaudio.transforms.TimeMasking(40)
        )
    elif name == "multi_mask":
        return MultiMask()
    elif name == "mel_warp":
        return MelWarp()
    elif name == "frame_drop":
        return FrameDrop()
    else:
        raise ValueError(f"Unknown augment: {name}")


