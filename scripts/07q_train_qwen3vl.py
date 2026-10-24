"""Run one declared Qwen3-VL QLoRA SFT variant on development images only."""
import argparse
from _bootstrap import PROJECT_ROOT
from uav_project.uav12_protocol import log_stage
from uav_project.uav12_qwen import train_qwen_lora

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["standard", "sqrt_class_balanced"], required=True)
    args = parser.parse_args()
    log_stage(f"train_qwen3vl_{args.variant}", lambda: train_qwen_lora(args.variant))
