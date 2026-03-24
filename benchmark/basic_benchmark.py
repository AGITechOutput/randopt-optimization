# basic_benchmark.py (开源演示版，仅验证动态分层噪声核心原理)
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.metrics import evaluate_accuracy  # 简化版精度计算工具

def add_dynamic_layer_noise(model, sigma_small=0.005, sigma_mid=0.01, sigma_large=0.02):
    """
    动态分层噪声注入（开源演示版）
    仅验证核心原理，不包含生产级并行、集成优化
    """
    num_layers = len(model.model.layers)
    s1, s2 = num_layers // 3, 2 * num_layers // 3
    layer_idx = 0
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            if layer_idx < s1:
                sigma = sigma_small
            elif layer_idx < s2:
                sigma = sigma_mid
            else:
                sigma = sigma_large
            param.data.add_(torch.randn_like(param) * sigma)
            layer_idx += 1
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-MoE")
    parser.add_argument("--noise_strategy", type=str, default="dynamic", choices=["fixed", "dynamic"])
    args = parser.parse_args()

    # 加载模型与tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    # 基线精度
    base_acc = evaluate_accuracy(model, tokenizer, dataset="gsm8k_subset")
    print(f"基线精度: {base_acc:.2f}%")

    # 应用噪声优化
    if args.noise_strategy == "dynamic":
        model = add_dynamic_layer_noise(model)
        print("已应用动态分层噪声")
    else:
        model = add_dynamic_layer_noise(model, sigma_small=0.02, sigma_mid=0.02, sigma_large=0.02)
        print("已应用固定强度噪声")

    # 优化后精度
    opt_acc = evaluate_accuracy(model, tokenizer, dataset="gsm8k_subset")
    print(f"优化后精度: {opt_acc:.2f}%")
    print(f"精度提升: {opt_acc - base_acc:.2f}%")

