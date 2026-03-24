# basic_benchmark.py (开源演示版，英伟达单卡最小化示例)
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

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

def evaluate_accuracy(model, tokenizer, dataset_subset):
    """
    简化版GSM8K精度计算（开源演示版）
    """
    correct = 0
    total = len(dataset_subset)
    for item in dataset_subset:
        inputs = tokenizer(item["question"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 简化匹配逻辑，实际完整评估见内部版
        if item["answer"] in pred:
            correct += 1
    return (correct / total) * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-MoE")
    parser.add_argument("--noise_strategy", type=str, default="dynamic", choices=["fixed", "dynamic"])
    args = parser.parse_args()

    # 加载模型与tokenizer
    print(f"[INFO] 加载模型: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # 加载GSM8K子集（100条样本，用于演示）
    print("[INFO] 加载GSM8K测试子集...")
    dataset = load_dataset("gsm8k", "main", split="test[:100]")

    # 基线精度
    base_acc = evaluate_accuracy(model, tokenizer, dataset)
    print(f"[INFO] 基线精度: {base_acc:.2f}%")

    # 应用噪声优化
    if args.noise_strategy == "dynamic":
        model = add_dynamic_layer_noise(model)
        print("[INFO] 已应用动态分层噪声")
    else:
        model = add_dynamic_layer_noise(model, sigma_small=0.02, sigma_mid=0.02, sigma_large=0.02)
        print("[INFO] 已应用固定强度噪声")

    # 优化后精度
    opt_acc = evaluate_accuracy(model, tokenizer, dataset)
    print(f"[INFO] 优化后精度: {opt_acc:.2f}%")
    print(f"[INFO] 精度提升: {opt_acc - base_acc:.2f}%")

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

