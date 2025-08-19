# fusion_factory.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class IdentityFusion(nn.Module):
    def forward(self, x1, x2):
        # 简单返回 x2，表示直接用频域特征（或主分支）
        return x2 if x1 is None else x1

class LightAttentionFusion(nn.Module):
    def __init__(self, time_dim, freq_dim, out_dim):
        super().__init__()
        self.q_proj = nn.Linear(freq_dim, out_dim)
        self.k_proj = nn.Linear(time_dim, out_dim)
        self.v_proj = nn.Linear(time_dim, out_dim)

    def forward(self, time_feat, freq_feat):
        # Q: [B, 1, D], K,V: [B, 1, D]
        Q = self.q_proj(freq_feat).unsqueeze(1)
        K = self.k_proj(time_feat).unsqueeze(1)
        V = self.v_proj(time_feat).unsqueeze(1)
        attn = F.softmax(Q @ K.transpose(-2, -1) / Q.shape[-1]**0.5, dim=-1)
        return (attn @ V).squeeze(1)  # 输出 [B, D]

class GatedSumFusion(nn.Module):
    def __init__(self, time_dim, freq_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(time_dim + freq_dim, freq_dim),
            nn.Sigmoid()
        )

    def forward(self, t, f):
        g = self.gate(torch.cat([t, f], dim=1))  # [B, D]
        return g * f + (1 - g) * t

class AddFusion(nn.Module):
    def __init__(self, time_dim, freq_dim):
        super().__init__()
        assert time_dim == freq_dim, "For AddFusion, time and freq dims must match"
    def forward(self, t, f):
        return t + f

def get_fusion_module(name, time_dim, freq_dim, out_dim):
    if name == "concat":
        return nn.Identity()
    elif name == "sum":
        return AddFusion(time_dim, freq_dim)
    elif name == "light_attn":
        return LightAttentionFusion(time_dim, freq_dim, out_dim)
    elif name == "gated_sum":
        return GatedSumFusion(time_dim, freq_dim)
    elif name == "none":
        return IdentityFusion()
    else:
        raise ValueError(f"Unknown fusion: {name}")
