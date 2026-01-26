import argparse
import json
import os
import shutil
from typing import Dict, Optional

import torch
import torch.nn as nn


def _is_target_module(module_name: str, module: nn.Module) -> bool:
    """
    与 gatedquantv2 对齐的目标模块选择规则：
    - 只作用于 OpenVLA 主干中的 Linear / Conv2d；
    - 包括 language_model.* 和 vision_backbone.*；
    - 显式排除 projector.* 以及动作头（action_head / lm_head）。
    """
    if module_name.startswith("projector.") or "action_head" in module_name or module_name.startswith("language_model.lm_head"):
        return False
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if module_name.startswith("language_model."):
            return True
        if module_name.startswith("vision_backbone."):
            return True
    return False


def _is_excluded_module(module_name: str) -> bool:
    # 显式排除 projector.* 与动作头（action_head / lm_head）
    return (
        module_name.startswith("projector.")
        or "action_head" in module_name
        or module_name.startswith("language_model.lm_head")
    )


def _load_gates(gate_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    加载 AutoQVLA 生成的 gates：
    - 支持 .pt（推荐）或 .json；
    - 约定为 {layer_name: LongTensor[num_output_channels]}，元素来自 {0,2,4,8,16}。
    """
    if gate_path.endswith(".pt"):
        raw = torch.load(gate_path, map_location=device)
    else:
        with open(gate_path, "r") as f:
            raw = json.load(f)

    out: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            t = torch.tensor(v, device=device, dtype=torch.int64)
        elif isinstance(v, torch.Tensor):
            t = v.to(device=device, dtype=torch.int64)
        else:
            t = torch.tensor(v, device=device, dtype=torch.int64)
        out[k] = t
    return out


@torch.no_grad()
def _fake_quantize_tensor_sym(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    简单的对称假量化：量化-反量化到给定位宽。
    - num_bits >= 16: 视为全精度，直接返回；
    - num_bits <= 0: 视为剪枝，返回全零。
    """
    if num_bits >= 16:
        return x
    if num_bits <= 0:
        return torch.zeros_like(x)
    qmax = (1 << (num_bits - 1)) - 1
    xmax = x.abs().max().clamp_min(1e-8)
    scale = xmax / qmax
    x_q = torch.clamp(torch.round(x / scale), min=-(qmax + 1), max=qmax)
    return x_q * scale


@torch.no_grad()
def _apply_weight_only_fake_quant(module: nn.Module, gates: torch.Tensor) -> None:
    """
    按输出通道 gates 位宽，对权重做一次性假量化：
    - Linear: weight 形状 [out_features, in_features]，按行处理；
    - Conv2d: weight 形状 [out_channels, in_channels, kH, kW]，按 out_channels 处理。

    激活保持全精度；不注册任何前向 hook。
    """
    if not hasattr(module, "weight"):
        return
    w = module.weight.data
    device = w.device
    g = gates.to(device=device, dtype=torch.int64)

    if isinstance(module, nn.Linear):
        out_channels = w.size(0)
        if g.numel() != out_channels:
            g = torch.full((out_channels,), int(g.median().item()), device=device, dtype=torch.int64)
        for i in range(out_channels):
            bw = int(g[i].item())
            if bw >= 16:
                continue
            if bw <= 0:
                w[i, :].zero_()
                continue
            row = w[i, :]
            w[i, :] = _fake_quantize_tensor_sym(row, bw)
    elif isinstance(module, nn.Conv2d):
        out_channels = w.size(0)
        if g.numel() != out_channels:
            g = torch.full((out_channels,), int(g.median().item()), device=device, dtype=torch.int64)
        for i in range(out_channels):
            bw = int(g[i].item())
            if bw >= 16:
                continue
            if bw <= 0:
                w[i, :, :, :].zero_()
                continue
            kernel = w[i, :, :, :]
            w[i, :, :, :] = _fake_quantize_tensor_sym(kernel, bw)


@torch.no_grad()
def inject_autoqvla_weight_fake_quant(
    model: nn.Module,
    gates_path: str,
    device: Optional[torch.device] = None,
) -> int:
    """
    在 OpenVLA 模型上注入 AutoQVLA 风格的“权重假量化”：

    - 不改变模型结构 / 不引入自定义 kernel；
    - 仅对目标模块的权重做一次性 per-channel 假量化；
    - gate 由 AutoQVLA 的敏感度 + gate 分配脚本离线生成。

    返回成功注入（即应用了假量化）的模块个数。
    """
    if device is None:
        device = next(iter(model.parameters())).device

    gates_map = _load_gates(gates_path, device=device)
    if len(gates_map) == 0:
        print("[AutoQVLA][fake-w] WARNING: gates map is empty, nothing to inject.")
        return 0

    injected = 0
    for name, module in model.named_modules():
        if _is_excluded_module(name):
            continue
        if not _is_target_module(name, module):
            continue

        g_tensor = None
        # 精确匹配
        if name in gates_map:
            g_tensor = gates_map[name]
        else:
            # 尝试去掉顶层前缀（language_model./vision_backbone.）
            if name.startswith("language_model."):
                short = name[len("language_model.") :]
                if short in gates_map:
                    g_tensor = gates_map[short]
            if g_tensor is None and name.startswith("vision_backbone."):
                short = name[len("vision_backbone.") :]
                if short in gates_map:
                    g_tensor = gates_map[short]

        if g_tensor is None:
            continue

        _apply_weight_only_fake_quant(module, gates=g_tensor)
        injected += 1

    print(f"[AutoQVLA][fake-w] applied weight-only fake quant to {injected} modules")
    return injected


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("f16", "fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def main():
    p = argparse.ArgumentParser(description="Inject weight-only fake quant (AutoQVLA) and save model.")
    p.add_argument("--pretrained_checkpoint", type=str, required=True, help="FP16 OpenVLA checkpoint dir")
    p.add_argument("--gates_path", type=str, required=True, help=".pt/.json gates (assign map)")
    p.add_argument("--out_dir", type=str, required=True, help="output dir to save fake-quantized model")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--dtype", type=str, default="bf16", help="bf16|f16|f32; ignored on cpu (uses f32)")
    args = p.parse_args()

    # 延迟注册 HF AutoClasses（与 openvla_utils 一致）
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    device = torch.device(args.device)
    torch_dtype = torch.float32 if device.type == "cpu" else _parse_dtype(args.dtype)
    attn_impl = "flash_attention_2" if device.type == "cuda" else "eager"

    os.makedirs(args.out_dir, exist_ok=True)

    model = AutoModelForVision2Seq.from_pretrained(
        args.pretrained_checkpoint,
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # 保留 norm_stats（用于动作反归一化）
    stats_path = os.path.join(args.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            model.norm_stats = json.load(f)

    injected = inject_autoqvla_weight_fake_quant(model, gates_path=args.gates_path, device=device)
    print(f"[AutoQVLA][fake-w] applied weight-only fake quant to {injected} modules")

    model.save_pretrained(args.out_dir)

    # 额外保存 / 同步 processor 与 tokenizer 相关文件，避免加载缺少配置
    try:
        processor = AutoProcessor.from_pretrained(args.pretrained_checkpoint, trust_remote_code=True)
        processor.save_pretrained(args.out_dir)
    except Exception as e:  # pragma: no cover - best effort
        print(f"[AutoQVLA][fake-w][warn] save processor failed: {e}")

    include_names = {
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_special_tokens_map.json",
        "special_tokens_map.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "processing_prismatic.py",
        "configuration_prismatic.py",
        "configuration.json",
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "processor.json",
        "dataset_statistics.json",
        "added_tokens.json",
        "README.md",
    }
    for name in include_names:
        src = os.path.join(args.pretrained_checkpoint, name)
        if os.path.isfile(src):
            dst = os.path.join(args.out_dir, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except Exception as e:  # pragma: no cover - best effort copy
                print(f"[AutoQVLA][fake-w][warn] copy {name} failed: {e}")

    print(f"[AutoQVLA][fake-w] saved model to {args.out_dir}")


if __name__ == "__main__":
    main()







