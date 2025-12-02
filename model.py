import math

import torch
import torch.nn as nn
import torchaudio
from torchvision import models
from torchvision.models import (
    RegNet_Y_400MF_Weights, ConvNeXt_Tiny_Weights,
    ShuffleNet_V2_X1_0_Weights, DenseNet201_Weights, MobileNet_V2_Weights,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights,
    EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, EfficientNet_B7_Weights
)
from torchvision.models.efficientnet import Conv2dNormActivation
from fusion_factory import get_fusion_module
from augment_factory import get_augment_module
from classifier_factory import get_classifier_module

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=256):
        super().__init__()
        self.nb_features = nb_features
        self.projection_matrix = nn.Parameter(torch.randn(dim_heads, nb_features))
        self.dim_heads = dim_heads

    def softmax_kernel(self, x):
        # Random feature approximation for softmax
        x_proj = torch.einsum('...id,df->...if', x, self.projection_matrix)  # [B, N, nb_features]
        return torch.exp(x_proj - x_proj.amax(dim=-1, keepdim=True)) / math.sqrt(self.nb_features)

    def forward(self, q, k, v):
        q_prime = self.softmax_kernel(q)
        k_prime = self.softmax_kernel(k)

        kv = torch.einsum('...nd,...ne->...de', k_prime, v)  # [nb_features, d]
        z = 1 / torch.einsum('...nd,...d->...n', q_prime, k_prime.sum(dim=-2) + 1e-6)[..., None]  # [B, N, 1]

        out = torch.einsum('...nd,...de->...ne', q_prime, kv)
        return out * z


class PerformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, mlp_ratio=2):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, heads * dim_head * 3)
        self.attn = FastAttention(dim_head)
        self.proj = nn.Linear(heads * dim_head, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.heads, self.dim_head).transpose(1, 2), qkv)

        attn_out = self.attn(q, k, v).transpose(1, 2).reshape(B, N, self.heads * self.dim_head)
        x = x + self.proj(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class Performer1D(nn.Module):
    def __init__(self, input_length=20635, in_channels=1, dim=128, depth=4, heads=4, dim_head=64, num_classes=None):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_channels, dim, kernel_size=16, stride=8, padding=4),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=8, stride=4, padding=2),
            nn.ReLU()
        )

        self.seq_len = input_length // 8 // 4  # downsample factor
        self.blocks = nn.Sequential(*[
            PerformerBlock(dim=dim, heads=heads, dim_head=dim_head)
            for _ in range(depth)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Identity() if num_classes is None else nn.Linear(dim, num_classes)

    def forward(self, x):  # x: [B, 1, T]
        x = self.patch_embed(x)             # [B, C, T']
        x = x.transpose(1, 2)               # [B, T', C]
        x = self.blocks(x)                  # [B, T', C]
        x = self.norm(x)                    # [B, T', C]
        x = x.transpose(1, 2)               # [B, C, T']
        x = self.pool(x).squeeze(-1)        # [B, C]
        return self.out_proj(x)             # [B, C] or [B, num_classes]
#1D ResNet 构建
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            BasicBlock1D(1, 64, stride=2),
            BasicBlock1D(64, 128, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):
        return self.layer(x).squeeze(-1)  # [B, 128]

class MLP1D(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=512, out_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(22050)  # ✅ 降采样到固定长度（适应 Linear 层）
        self.model = nn.Sequential(
            nn.Flatten(),                      # [B, 1, 22050] → [B, 22050]
            nn.Linear(22050, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), nn.ReLU(),
        )

    def forward(self, x):  # x: [B, 1, T]
        if x.ndim == 2:  # [B, T] → [B, 1, T]
            x = x.unsqueeze(1)
        x = self.pool(x)  # [B, 1, T] → [B, 1, 22050]
        return self.model(x)

class Transformer1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [B, 1, T]
        x = self.proj(x).permute(2, 0, 1)  # [T, B, C]
        x = self.encoder(x).permute(1, 2, 0)  # [B, C, T]
        return self.pool(x).squeeze(-1)  # [B, C]


# 加载主干网络

def get_backbone(name, pretrained):
    if name == "none":
        return nn.Identity()
    weights_map = {
        "efficientnet_b0": EfficientNet_B0_Weights,
        "efficientnet_b1": EfficientNet_B1_Weights,
        "efficientnet_b2": EfficientNet_B2_Weights,
        "efficientnet_b3": EfficientNet_B3_Weights,
        "efficientnet_b4": EfficientNet_B4_Weights,
        "efficientnet_b5": EfficientNet_B5_Weights,
        "efficientnet_b6": EfficientNet_B6_Weights,
        "efficientnet_b7": EfficientNet_B7_Weights,
        "resnet50": models.ResNet50_Weights,
        "regnet": RegNet_Y_400MF_Weights,
        "convnext": ConvNeXt_Tiny_Weights,
        "shufflenetv2": ShuffleNet_V2_X1_0_Weights,
        "densenet201": DenseNet201_Weights,
        "mobilenetv2": MobileNet_V2_Weights
    }

    model_fn_map = {
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "efficientnet_b2": models.efficientnet_b2,
        "efficientnet_b3": models.efficientnet_b3,
        "efficientnet_b4": models.efficientnet_b4,
        "efficientnet_b5": models.efficientnet_b5,
        "efficientnet_b6": models.efficientnet_b6,
        "efficientnet_b7": models.efficientnet_b7,
        "resnet50": models.resnet50,
        "regnet": models.regnet_y_400mf,
        "convnext": models.convnext_tiny,
        "shufflenetv2": models.shufflenet_v2_x1_0,
        "densenet201": models.densenet201,
        "mobilenetv2": models.mobilenet_v2,
    }

    if name not in model_fn_map:
        raise ValueError(f"Unsupported backbone: {name}")

    weights = None
    if pretrained == "imagenet_v1":
        weights = weights_map[name].IMAGENET1K_V1
    elif pretrained == "imagenet_v2" and hasattr(weights_map[name], "IMAGENET1K_V2"):
        weights = weights_map[name].IMAGENET1K_V2
    elif pretrained == "none":
        weights = None

    return model_fn_map[name](weights=weights)


# 主模型结构

class DualBranchFusionNet(nn.Module):
    def __init__(self, backbone_name, fusion_mode, classifier_mode, time_branch_type ,pretrained, augment_name, num_classes):
        super().__init__()
        self.fusion_mode = fusion_mode

        '''
        #时域分支
        self.time_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 输出 [B, 128, 1]
        )
        '''
        # 时域分支
        if time_branch_type == "cnn":
            self.time_branch = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=9, stride=2, padding=4),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
        elif time_branch_type == "resnet1d":
            self.time_branch = ResNet1D()
        elif time_branch_type == "transformer1d":
            self.time_branch = Transformer1D()
        elif time_branch_type == "mlp1d":
            self.time_branch = MLP1D()
        elif time_branch_type == "none":
            self.time_branch = nn.Identity()
        elif time_branch_type == "Performer1D":
            self.time_branch = Performer1D()
        else:
            raise ValueError(f"Unsupported time_branch_type: {time_branch_type}")


        # 频谱图变换 + 增强
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100, n_fft=1024, hop_length=256, n_mels=128)
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.augment_name = augment_name
        self.spec_aug = get_augment_module(augment_name)

        # 主干网络（频域）
        self.backbone = get_backbone(backbone_name, pretrained)
        # if "efficientnet" in backbone_name:
        #     self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        #     feat_dim = self.backbone.classifier[1].in_features
        #     self.backbone.classifier = nn.Identity()
        if backbone_name in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]:
            self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif backbone_name in ["efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6",
                               "efficientnet_b7"]:
            firstconv = Conv2dNormActivation(
                in_channels=1,
                out_channels=self.backbone.features[0][0].out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_layer=torch.nn.BatchNorm2d,
                activation_layer=torch.nn.SiLU
            )
            self.backbone.features[0][0] = firstconv
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif "resnet" in backbone_name:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            self.backbone.classifier = nn.Identity()
            feat_dim = 1280  # fallback

      
        if fusion_mode in ["sum", "gated_sum", "light_attn", "residual_fuse", "cross_mul"]:
            self.feat_projector = nn.Linear(feat_dim, 128)
            feat_dim = 128
        else:
            self.feat_projector = nn.Identity()

        # 融合模块
        if fusion_mode == "concat":
            self.fusion = nn.Identity()
            fusion_out = 128 + feat_dim
        else:
            self.fusion = get_fusion_module(fusion_mode, 128, feat_dim, 128)
            fusion_out = 128

        # 分类器（非创新，统一使用 MLP）
        self.classifier = get_classifier_module(classifier_mode, fusion_out, num_classes)

    def forward(self, x):

        # 
        # x_time = self.time_branch(x).squeeze(-1)  # [B, 128]
        # 
        # x_freq = self.db(self.mel(x.squeeze(1)))     # [B, 128, T]
        #
        # if self.augment_name in ["frame_drop", "mel_warp"]:
        #     x_freq = self.spec_aug(x_freq)
        # else:
        #     x_freq = self.spec_aug(x_freq).unsqueeze(1)  # [B, 1, 128, T0]
        # x_feat = self.backbone(x_freq)               # [B, feat_dim]
        # x_feat = self.feat_projector(x_feat)         

        #
        # if self.fusion_mode == "concat":
        #     x_all = torch.cat([x_time, x_feat], dim=1)   # [B, 128 + feat_dim]
        # else:
        #     x_all = self.fusion(x_time, x_feat)          # [B, 128]
        #
        # return self.classifier(x_all)
        #  时域分支
        if isinstance(self.time_branch, nn.Identity):
            x_time = None
        else:
            x_time = self.time_branch(x).squeeze(-1)  # [B, 128]

        #  频域分支 
        if isinstance(self.backbone, nn.Identity):
            x_feat = None
        else:
            x_freq = self.db(self.mel(x.squeeze(1)))  # [B, 128, T]
            if self.augment_name in ["frame_drop", "mel_warp"]:
                x_freq = self.spec_aug(x_freq)
            else:
                x_freq = self.spec_aug(x_freq).unsqueeze(1)  # [B, 1, 128, T0]
            x_feat = self.backbone(x_freq)  # [B, feat_dim]
            x_feat = self.feat_projector(x_feat)

        #  融合 
        if x_time is None:
            x_all = x_feat
        elif x_feat is None:
            x_all = x_time
        else:
            if self.fusion_mode == "concat":
                x_all = torch.cat([x_time, x_feat], dim=1)
            else:
                x_all = self.fusion(x_time, x_feat)

        return self.classifier(x_all)


# 构建模型函数

def build_model(config, num_classes):
    return DualBranchFusionNet(
        backbone_name=config["backbone"],
        fusion_mode=config["fusion"],
        classifier_mode=config["classifier"],
        pretrained=config["pretrained"],
        augment_name=config["augment"],
        num_classes=num_classes,
        time_branch_type=config["backbone1"]
    )

