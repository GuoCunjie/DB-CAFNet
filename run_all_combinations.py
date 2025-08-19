# run_all_combinations.py（用于运行 A0~A5 消融实验）
import os
import subprocess
import json

# 加载所有 ablation 配置
with open("list.json") as f:
    all_configs = json.load(f)

# 日志输出路径
log_path = "ablation_results.csv"
if not os.path.exists(log_path):
    with open(log_path, "w") as f:
        f.write("name,fusion,classifier,augment,backbone,disable_time,disable_freq,accuracy,f1_macro,mean_ap,flops,params\n")

for config in all_configs:
    name = config.get("name", "unnamed")
    fusion = config.get("fusion", "concat")
    classifier = config.get("classifier", "channel_mask")
    augment = config.get("augment", "specaugment")
    backbone = config.get("backbone", "efficientnet_b0")
    disable_time = config.get("disable_time_branch", False)
    disable_freq = config.get("disable_freq_branch", False)
    backbone1=config.get("backbone1", "cnn")
    # 实验路径与指标文件路径
    exp_name = name
    metrics_path = f"run/{exp_name}/metrics/metrics.json"

    if os.path.exists(metrics_path):
        print(f"✅ Skip: {exp_name}")
        continue

    print(f"🚀 Running: {exp_name}")

    # 写入配置文件（供 main.py 使用）
    with open("ablation_config.json", "w") as f:
        json.dump(config, f, indent=2)

    try:
        subprocess.run(["python", "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {exp_name}: {e}")
        continue

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        with open(log_path, "a") as f:
            f.write(f"{name},{fusion},{classifier},{augment},{backbone},{disable_time},{disable_freq},{backbone1}"
                    f"{m['accuracy']:.4f},{m['f1_macro']:.4f},{m['mean_ap']:.4f},{m['flops_gflops']},{m['params_million']}\n")
