# main.py
import os
import json
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.backends import cudnn

from dataset import InsectDataset
from model import build_model
from train import train_model

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False

# ✅ 支持从 JSON 读取配置（用于 run_all_combinations.py）
if os.path.exists("ablation_config.json"):
    with open("ablation_config.json") as f:
        ablation_config = json.load(f)
else:
    ablation_config = {

        "fusion": "concat",                # fusion_factory.py
        "augment": "specaugment",          # augment_factory.py
        "loss": "label_smoothing",
        "optimizer": "adamw_lookahead",
        "scheduler": "onecycle",
        "pretrained": "imagenet_v1",
        "backbone": "efficientnet_b0",
        "classifier": "channel_mask",
        "name":"test",# classifier_factory.py
        "backbone1": "cnn",
        "data_root": "data/set"
    }

if __name__ == '__main__':
    # 基础设置
  #  dataset_root = "data/set"
   # dataset_root = "data/set47"
    dataset_root = ablation_config["data_root"]
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_set = InsectDataset(dataset_root, mode='train')
    test_set = InsectDataset(dataset_root, mode='test')


    num_classes = len(set(train_set.labels))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)

    # 构建模型
    model = build_model(ablation_config, num_classes).to(device)

    # 生成实验路径
    #exp_name = f"{ablation_config['backbone']}-{ablation_config['fusion']}-{ablation_config['augment']}-" \
               #f"{ablation_config['loss']}-{ablation_config['optimizer']}-{ablation_config['scheduler']}-" \
               #f"{ablation_config['pretrained']}-{ablation_config['classifier']}"
    exp_name = ablation_config["name"]

    # 训练模型
    train_model(model, train_loader, test_loader, num_classes, device, ablation_config, exp_name)
