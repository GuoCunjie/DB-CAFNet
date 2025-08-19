# single_branch.py
# =========================================================
# 单文件可独立运行：支持单时域(time_only)与单频域(freq_only)训练
# - 数据目录结构：data/set/{train,test}/class/*.wav
# - 指标与可视化保存路径：run/<exp_name>/
# - 与原项目 train.py 的评估口径保持一致
# =========================================================

import os
import json
import random
import argparse
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.backends import cudnn

# ========= 复用你项目中的模块（分类器与增强） =========
from classifier_factory import get_classifier_module   # 保持与原项目一致
from augment_factory import get_augment_module         # 保持与原项目一致

# ========= 训练与评估（直接复用原有实现，保证口径一致） =========
from train import train_model  # 会按原样保存 metrics/plots/logs 等全部产物


# ======================
# 数据集（保持与 dataset.py 一致）
# ======================
class InsectDataset(Dataset):
    """
    与你当前 dataset.py 逻辑一致：
    - 统一采样率
    - 单声道
    - 短则重复填充，长则随机裁剪
    """
    def __init__(self, root_dir, mode='train', sample_rate=44100, duration=4.68):
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration)
        self.data, self.labels, self.label_map = [], [], {}
        path = os.path.join(root_dir, mode)
        for idx, folder in enumerate(sorted(os.listdir(path))):
            self.label_map[folder] = idx
            for file in os.listdir(os.path.join(path, folder)):
                if file.endswith(".wav"):
                    self.data.append(os.path.join(path, folder, file))
                    self.labels.append(idx)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.data[idx])
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        # 单声道
        waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
        # 对齐长度
        if waveform.shape[1] < self.target_len:
            repeat = self.target_len // waveform.shape[1] + 1
            waveform = waveform.repeat(1, repeat)[:, :self.target_len]
        else:
            start = random.randint(0, waveform.shape[1] - self.target_len)
            waveform = waveform[:, start:start + self.target_len]
        return waveform, self.labels[idx]


# ======================
# Backbone 构建（频域用）
# ======================
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    RegNet_Y_400MF_Weights, ConvNeXt_Tiny_Weights,
    ShuffleNet_V2_X1_0_Weights, DenseNet201_Weights, MobileNet_V2_Weights
)

def get_backbone(name: str, pretrained: str):
    """
    与你原 model.py 的函数对齐：支持 imagenet_v1 / imagenet_v2 / none
    """
    weights_map = {
        "efficientnet_b0": EfficientNet_B0_Weights,
        "efficientnet_b1": EfficientNet_B1_Weights,
        "efficientnet_b2": EfficientNet_B2_Weights,
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


# ======================
# 单分支模型
# ======================
class TimeOnlyNet(nn.Module):
    """
    时域单分支：沿用你原本 DualBranchFusionNet 的 time_branch + 可选分类器
    """
    def __init__(self, classifier_mode: str, num_classes: int):
        super().__init__()
        self.time_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = get_classifier_module(classifier_mode, 128, num_classes)

    def forward(self, x: torch.Tensor):
        x_time = self.time_branch(x).squeeze(-1)  # [B, 128]
        return self.classifier(x_time)


class FreqOnlyNet(nn.Module):
    """
    频域单分支：Mel -> dB -> (可选增强) -> 1 通道输入主干 -> GAP 后接分类器
    主干输入改为单通道，输出特征维度自动适配
    """
    def __init__(self, backbone_name: str, classifier_mode: str, pretrained: str,
                 augment_name: str, num_classes: int):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100, n_fft=1024, hop_length=256, n_mels=128
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.spec_aug = get_augment_module(augment_name)

        self.backbone = get_backbone(backbone_name, pretrained)
        # 统一改为单通道输入，并截取主干的分类头前的特征维度
        if "efficientnet" in backbone_name:
            self.backbone.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif "resnet" in backbone_name:
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif "regnet" in backbone_name:
            self.backbone.stem[0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif "convnext" in backbone_name:
            self.backbone.features[0][0] = nn.Conv2d(  # stem
                1, 96, kernel_size=4, stride=4, bias=False
            )
            feat_dim = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        elif "mobilenetv2" in backbone_name:
            # MobileNetV2 第一层 Conv2d 改为 1 通道
            first_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                1, first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False
            )
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif "densenet201" in backbone_name:
            self.backbone.features.conv0 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            feat_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif "shufflenetv2" in backbone_name:
            self.backbone.conv1[0] = nn.Conv2d(
                1, 24, kernel_size=3, stride=2, padding=1, bias=False
            )
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            # 兜底：假定 1280 维
            feat_dim = 1280
            if hasattr(self.backbone, "classifier"):
                self.backbone.classifier = nn.Identity()

        self.classifier = get_classifier_module(classifier_mode, feat_dim, num_classes)

    def forward(self, x: torch.Tensor):
        # x: [B, 1, T]
        x_freq = self.db(self.mel(x.squeeze(1)))    # [B, n_mels, time]
        x_freq = self.spec_aug(x_freq).unsqueeze(1) # [B, 1, n_mels, time]
        x_feat = self.backbone(x_freq)              # [B, C]
        return self.classifier(x_feat)


# ======================
# 构建模型
# ======================
def build_single_branch_model(config: dict, num_classes: int) -> nn.Module:
    mode = config.get("mode", "time_only")
    if mode == "time_only":
        return TimeOnlyNet(config["classifier"], num_classes)
    elif mode == "freq_only":
        return FreqOnlyNet(
            backbone_name=config["backbone"],
            classifier_mode=config["classifier"],
            pretrained=config["pretrained"],
            augment_name=config["augment"],
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'time_only' or 'freq_only'.")


# ======================
# 入口：训练
# ======================
def parse_args():
    p = argparse.ArgumentParser(description="Single-Branch Trainer (Time-Only / Freq-Only)")
    # 运行模式
    p.add_argument("--mode", type=str, default="freq_only",
                   choices=["time_only", "freq_only"],
                   help="选择单时域或单频域训练")
    # 数据与训练
    p.add_argument("--data_root", type=str, default="data/set", help="数据根目录")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    # 模型/策略
    p.add_argument("--backbone", type=str, default="efficientnet_b1",
                   choices=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
                            "resnet50", "regnet", "convnext", "shufflenetv2",
                            "densenet201", "mobilenetv2"])
    p.add_argument("--pretrained", type=str, default="imagenet_v1",
                   choices=["imagenet_v1", "imagenet_v2", "none"])
    p.add_argument("--augment", type=str, default="specaugment",
                   choices=["none", "specaugment", "strong_freq", "strong_time", "multi_mask", "mel_warp", "frame_drop"])
    p.add_argument("--classifier", type=str, default="channel_mask",
                   choices=[
                       "default","residual_gate","dense_shrink","mlp_mix",
                       "channel_mask","token_excitation","bi_proj_residual",
                       "dual_gate","norm_adaptive","GAP","attentive_mlp",
                       "resmlp","TMRC","TGR-Classifier","SSC"
                   ])
    # 优化与调度在 train.py 内固定为 ablation_config 的字段；这里仅写入配置文件
    p.add_argument("--optimizer", type=str, default="adamw_lookahead",
                   choices=["adamw", "adamw_lookahead", "adam", "adam_lookahead", "sgd", "sgd_lookahead"])
    p.add_argument("--scheduler", type=str, default="onecycle",
                   choices=["onecycle", "none"])
    # 其他
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # 固定随机种子（与原 main.py 一致）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    train_set = InsectDataset(args.data_root, mode='train')
    test_set  = InsectDataset(args.data_root, mode='test')
    num_classes = len(set(train_set.labels))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, num_workers=args.num_workers)

    # 组装配置（与原项目字段保持一致）
    ablation_config = {
        "mode": args.mode,                  # 新增字段，区分单分支
        "fusion": "none",                   # 单分支不需要融合，但为兼容性保留
        "augment": args.augment,
        "loss": "label_smoothing",
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "pretrained": args.pretrained,
        "backbone": args.backbone,
        "classifier": args.classifier
    }

    # 构建模型
    model = build_single_branch_model(ablation_config, num_classes).to(device)

    # 生成实验名（含单分支模式）
    exp_name = f"{args.mode}-{ablation_config['backbone']}-{ablation_config['augment']}-" \
               f"{ablation_config['loss']}-{ablation_config['optimizer']}-{ablation_config['scheduler']}-" \
               f"{ablation_config['pretrained']}-{ablation_config['classifier']}"

    # 开训（复用原 train_model，保证与原来完全一致的保存与指标）
    train_model(model, train_loader, test_loader, num_classes, device, ablation_config, exp_name)


if __name__ == "__main__":
    main()
