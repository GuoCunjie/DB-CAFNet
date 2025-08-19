Thank you for clicking into my project
You can download the dataset：
  InsectSet32: https://zenodo.org/records/7072196
  InsectSet47: https://zenodo.org/records/8252141
  InsectSet66: https://zenodo.org/records/8252141
  ESC-50: https://github.com/karoldvl/ESC-50
  FSDKaggle2018: https://zenodo.org/record/2552860
Before that, you need to divide the dataset into a format supported by the code：set/train/class/XXX.wav,set/test/class/XXX.wav,set/val/class/XXX.wav
At first,If you want to train a single combination, run main.py，The steps are as follows：
  Method one：Create ablation_config.json and write:
    {
    "fusion": "concat",#["concat", "light_attn", "gated_sum", "sum"]
    "augment": "specaugment",#["none", "specaugment", "multi_mask", "mel_warp", "frame_drop"]
    "loss": "label_smoothing",
    "optimizer": "adamw_lookahead",
    "scheduler": "onecycle",
    "pretrained": "imagenet_v1",#["imagenet_v1", "imagenet_v2", "none"]
    "backbone": "efficientnet_b0",#[efficientnet_b0-b7]
    "classifier": "channelmask",#[linear,dropblock1d,dropout,channel_mask]
    "name": "name",
    "backbone1": "cnn",#[cnn,mlp,resnet1d,performer1d]
    "data_root": "data/set"#different data
    }
    then run main.py
  Method two:Directly find the configuration option and select configuration, then run the file. Make sure that the file of method 1 does not exist
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
  Then,If you want batch training, create list.json and write [{configuration 1}, {configuration 2}, ...],then run run_all_combinations.py
  At last,If you want to run a single branch, select single.py and select time_only or freq_only by default in p.add_argument("--mode", type=str, default="freq_only",
    choices=["time_only", "freq_only"],
    help="Select single time domain or single frequency domain training"), 
    and then train.
  
