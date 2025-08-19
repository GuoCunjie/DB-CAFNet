
import torch
import torch.nn as nn
import torch.nn.functional as F
#不同正则化
class DropBlock1DClassifier(nn.Module):
    """
    1D向量上的“连续段失活”——把向量看作1D feature map，在训练期把一段连续通道置零。
    这是适配向量空间的DropBlock思想（空间上是连续块、这里是连续通道段）。
    推理期零开销。
    """
    def __init__(self, in_dim: int, num_classes: int, block_frac: float = 0.2, n_blocks: int = 1):
        super().__init__()
        assert 0.0 <= block_frac < 1.0
        self.in_dim = in_dim
        self.block_len = max(1, int(round(in_dim * block_frac)))
        self.n_blocks = max(1, int(n_blocks))
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.block_len > 0:
            B, C = x.shape
            for b in range(B):
                for _ in range(self.n_blocks):
                    if C <= self.block_len:
                        x[b] = 0
                    else:
                        start = torch.randint(0, C - self.block_len + 1, (1,), device=x.device).item()
                        x[b, start:start + self.block_len] = 0
        return self.fc(x)

class IdentityClassifier(nn.Module):
    """无正则基线：单层线性分类器"""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class DropoutClassifier(nn.Module):
    """
    元素级Dropout：对向量每个元素独立失活（经典Dropout）。
    训练期有效；推理期自动关闭（PyTorch行为）。
    """
    def __init__(self, in_dim: int, num_classes: int, p: float = 0.5):
        super().__init__()
        self.drop = nn.Dropout(p=p)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        return self.fc(x)
#不同分类器
class DropPathMLPClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.proj_res = nn.Linear(in_dim, 256)  # residual projection
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        residual = self.proj_res(x)  # [B, 256]
        x = self.fc1(x)              # [B, 256]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)              # [B, 256]
        x = x + residual             # residual connection
        return self.out(x)
class ChannelDropClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, drop_rate=0.3):
        super().__init__()
        self.drop_rate = drop_rate
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        if self.training:
            mask = torch.rand_like(x) > self.drop_rate
            x = x * mask
        return self.fc(x)


def get_classifier_module(name, in_dim, num_classes):
    if name == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    elif name == "channel_mask":
        return ChannelDropClassifier(in_dim, num_classes)
    elif name == "resmlp":
        return DropPathMLPClassifier(in_dim, num_classes)
    elif name == "GAP":
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes)
        )
    elif name == "attentive_mlp":
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, num_classes)
        )
    elif name == "linear":
        return IdentityClassifier(in_dim, num_classes)
    elif name == "dropout":
        return DropoutClassifier(in_dim, num_classes, p=0.5)
    elif name == "dropblock1d":
        return DropBlock1DClassifier(in_dim, num_classes, block_frac=0.2, n_blocks=1)
    else:
        raise ValueError(f"Unknown classifier: {name}")
