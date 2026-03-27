# basic_benchmark_nvidia.py（开源演示版，英伟达单卡优化版）
# 版本：v3.2-final（英伟达单卡专属｜修复tokenizer重复加载｜强化答案匹配）
# 功能：动态分层噪声验证 + 多轮统计、可视化、详细日志
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import matplotlib.pyplot as plt
import numpy as np
import re

def add_noise_inplace(model, sigma_small=0.005, sigma_mid=0.01, sigma_large=0.02, verbose=False):
    num_layers = len(model.model.layers)
    s1, s2 = num_layers // 3, 2 * num_layers // 3
    if verbose:
        print("[VERBOSE] 开始注入动态分层噪声...")
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx < s1:
            sigma = sigma_small
        elif layer_idx < s2:
            sigma = sigma_mid
        else:
            sigma = sigma_large
        if verbose and (layer_idx < 5 or layer_idx > num_layers-5):
            print(f"[VERBOSE] Layer {layer_idx}: σ={sigma:.4f}")
        for name, param in layer.named_parameters():
            if "weight" in name and param.requires_grad:
                weight_fp32 = param.data.float()
                noise = torch.randn_like(weight_fp32) * sigma
                param.data = (weight_fp32 + noise).to(param.dtype)
    return model

def extract_answer(pred: str) -> str:
    """从模型输出中提取最终答案（GSM8K格式）"""
    # 尝试匹配 "The answer is X." 或末尾数字
    match = re.search(r'The answer is (\d+)\.', pred)
    if match:
        return match.group(1)
    # 否则取最后一串数字
    numbers = re.findall(r'\d+', pred)
    return numbers[-1] if numbers else ""

def evaluate_accuracy_batched(model, tokenizer, dataset_subset, batch_size=8):
    model.eval()
    device = model.device
    correct = 0
    total = len(dataset_subset)
    for i in range(0, total, batch_size):
        batch_data = dataset_subset[i:i+batch_size]
        questions = [item["question"] for item in batch_data]
        answers = [item["answer"] for item in batch_data]
        inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for pred, answer in zip(preds, answers):
            pred_ans = extract_answer(pred)
            if pred_ans == answer.strip():
                correct += 1
    return round((correct / total) * 100, 2)

def visualize_noise(model_name, sigma_small, sigma_mid, sigma_large):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    num_layers = len(model.model.layers)
    s1, s2 = num_layers // 3, 2 * num_layers // 3
    strengths = []
    for i in range(num_layers):
        if i < s1:
            strengths.append(sigma_small)
        elif i < s2:
            strengths.append(sigma_mid)
        else:
            strengths.append(sigma_large)
    plt.figure(figsize=(10,4))
    plt.bar(range(num_layers), strengths, color='blue')
    plt.xlabel('Layer Index')
    plt.ylabel('Noise Standard Deviation')
    plt.title('Dynamic Layer Noise Strength')
    plt.savefig('noise_distribution.png')
    print("[INFO] 噪声分布图已保存为 noise_distribution.png")
    del model
    torch.cuda.empty_cache()

def run_multi_rounds(model_name, tokenizer, dataset, noise_strategy, batch_size, sigma, repeat, save_result):
    gains = []
    # 提前加载 tokenizer（避免每轮重复 I/O）
    tokenizer_local = AutoTokenizer.from_pretrained(model_name)
    tokenizer_local.pad_token = tokenizer_local.eos_token
    
    for r in range(repeat):
        print(f"[INFO] 第 {r+1}/{repeat} 轮")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        base_acc = evaluate_accuracy_batched(model, tokenizer_local, dataset, batch_size)
        if noise_strategy == "dynamic":
            model = add_noise_inplace(model, sigma[0], sigma[1], sigma[2])
        else:
            model = add_noise_inplace(model, 0.02, 0.02, 0.02)
        opt_acc = evaluate_accuracy_batched(model, tokenizer_local, dataset, batch_size)
        gain = opt_acc - base_acc
        gains.append(gain)
        print(f"[INFO] 本轮提升: {gain:.2f}%")
        del model
        torch.cuda.empty_cache()
    mean_gain = sum(gains)/len(gains)
    std_gain = (sum((g-mean_gain)**2 for g in gains)/len(gains))**0.5
    print(f"\n[RESULT] 重复{repeat}轮平均提升: {mean_gain:.2f}% (±{std_gain:.3f})")
    if save_result:
        with open("repeat_results.json", "w") as f:
            json.dump({"gains": gains, "mean": mean_gain, "std": std_gain}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-MoE")
    parser.add_argument("--noise_strategy", type=str, default="dynamic", choices=["fixed","dynamic"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sigma_small", type=float, default=0.005)
    parser.add_argument("--sigma_mid", type=float, default=0.01)
    parser.add_argument("--sigma_large", type=float, default=0.02)
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="多轮验证次数")
    parser.add_argument("--visualize", action="store_true", help="生成噪声分布图")
    parser.add_argument("--verbose", action="store_true", help="打印详细日志")
    args = parser.parse_args()

    if args.visualize:
        visualize_noise(args.model, args.sigma_small, args.sigma_mid, args.sigma_large)
        exit(0)

    print("[INFO] 加载GSM8K测试子集（100条）...")
    dataset = load_dataset("gsm8k", "main", split="test[:100]")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    if args.repeat > 1:
        run_multi_rounds(args.model, tokenizer, dataset, args.noise_strategy,
                         args.batch_size, (args.sigma_small, args.sigma_mid, args.sigma_large),
                         args.repeat, args.save_result)
        exit(0)

    # 单次运行
    print(f"[INFO] 加载模型: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print("[INFO] 计算基线精度...")
    base_acc = evaluate_accuracy_batched(model, tokenizer, dataset, args.batch_size)
    print(f"[RESULT] 基线精度: {base_acc:.2f}%")
    if args.noise_strategy == "dynamic":
        print(f"[INFO] 应用动态分层噪声（σ={args.sigma_small}/{args.sigma_mid}/{args.sigma_large}）")
        model = add_noise_inplace(model, args.sigma_small, args.sigma_mid, args.sigma_large, args.verbose)
    else:
        print("[INFO] 应用固定强度噪声（σ=0.02）")
        model = add_noise_inplace(model, 0.02, 0.02, 0.02, args.verbose)
    print("[INFO] 计算优化后精度...")
    opt_acc = evaluate_accuracy_batched(model, tokenizer, dataset, args.batch_size)
    print(f"[RESULT] 优化后精度: {opt_acc:.2f}%")
    print(f"[RESULT] 精度提升: {opt_acc - base_acc:.2
