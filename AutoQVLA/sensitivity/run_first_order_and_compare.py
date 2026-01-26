import argparse
import os
from typing import Dict, Tuple, Iterable

import torch
import torch.nn as nn

# 允许脚本直接运行：当作为脚本执行时，手动把 AutoQVLA 上级目录加入 sys.path
import sys as _sys
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)

# 引入 mydesk/gatedquantv2 中的一阶代理实现，替换原 AutoQVLA 的敏感度计算
try:
    from mydesk.gatedquantv2.sensitivity.compute_first_order_proxy import (  # type: ignore
        compute_first_order_proxy,
        _build_calib_iter,
    )
except Exception:
    # 兼容相对路径（当作为包运行时）
    _repo_root = os.path.dirname(_parent)  # .../openvla
    _maybe_mydesk = os.path.join(_repo_root, "mydesk")
    if _maybe_mydesk not in _sys.path:
        _sys.path.insert(0, _maybe_mydesk)
    from gatedquantv2.sensitivity.compute_first_order_proxy import (  # type: ignore
        compute_first_order_proxy,
        _build_calib_iter,
    )


def load_model_and_processor(ckpt: str, device: torch.device) -> Tuple[nn.Module, object]:
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        ckpt,
        attn_implementation=("flash_attention_2" if device.type == "cuda" else "eager"),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    return model, processor


def _quantization_row_noise_l2(
    w_flat: torch.Tensor,
    bits: Iterable[int],
) -> Dict[int, torch.Tensor]:
    """
    给定按 [C_out, D] 展平后的权重矩阵 w_flat，对每个比特 b 计算：
        ΔW_b = Q_b(W) - W
        noise_b[c] = ||ΔW_b[c, :]||_2

    这里的 Q_b 为对称均匀量化（行级 scale），用于近似“该通道在比特 b 下
    对输出特征引入的扰动幅度 ||ΔX_{l,c}(b)||”。
    """
    w = w_flat.detach().float()
    C, D = w.shape
    out: Dict[int, torch.Tensor] = {}
    for b in bits:
        if b <= 0:
            # 0-bit：视为整行权重被置零，对应 ΔW = -W
            delta = -w
        else:
            qmax = (1 << (b - 1)) - 1  # 2-bit->1, 4-bit->7, 8-bit->127
            max_abs = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
            scale = max_abs / float(qmax)
            q = torch.round(w / scale).clamp_(-qmax - 1, qmax)
            w_q = q * scale
            delta = w_q - w
        # L2 范数作为扰动尺度近似
        noise = delta.norm(p=2, dim=1)  # [C]
        out[int(b)] = noise
    return out


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().float()
    y = y.flatten().float()
    assert x.numel() == y.numel()
    n = x.numel()
    if n <= 1:
        return float("nan")
    rx = torch.argsort(torch.argsort(x))
    ry = torch.argsort(torch.argsort(y))
    rx = rx.float()
    ry = ry.float()
    rx = (rx - rx.mean()) / (rx.std() + 1e-8)
    ry = (ry - ry.mean()) / (ry.std() + 1e-8)
    return float((rx * ry).mean().item())


def compare_rank(first_order_path: str, cum_path: str, bit: int) -> Dict[str, float]:
    """
    first_order_path: torch.save 的一阶敏感度（支持旧格式 Tensor[C] 或新格式含 fo_score）
    cum_path: action_mse/compute_action_sensitivity.py 生成的 {layer: {'mse_2',...}}
    """
    first = torch.load(first_order_path, map_location="cpu")
    cum = torch.load(cum_path, map_location="cpu")
    key = f"mse_{bit}"

    per_layer: Dict[str, float] = {}
    xs = []
    ys = []
    for name, s in first.items():
        if name not in cum:
            continue
        d = cum[name]
        if key not in d:
            continue
        v = d[key]
        if isinstance(v, torch.Tensor):
            e = v.detach().float()
        else:
            e = torch.tensor(v, dtype=torch.float32)
        # 兼容新格式：取 fo_score
        if isinstance(s, dict):
            if "fo_score" not in s:
                continue
            s_val = s["fo_score"]
        else:
            s_val = s
        s_t = s_val.detach().float() if torch.is_tensor(s_val) else torch.tensor(s_val, dtype=torch.float32)
        if s_t.numel() != e.numel():
            continue
        c = _spearman_corr(s_t, e)
        per_layer[name] = c
        xs.append(s_t)
        ys.append(e)

    if xs:
        x_full = torch.cat(xs)
        y_full = torch.cat(ys)
        global_corr = _spearman_corr(x_full, y_full)
    else:
        global_corr = float("nan")

    per_layer["__global__"] = global_corr
    return per_layer


def save_first_order_scores(
    ckpt: str,
    calib_jsonl: str,
    out_path: str,
    device: torch.device,
    max_samples: int = 64,
    bits: Tuple[int, ...] = (0, 2, 4, 8),
    max_batches: int = None,
) -> None:
    model, processor = load_model_and_processor(ckpt, device)
    dl_processor, dl = _build_calib_iter(calib_jsonl, ckpt, device, max_samples)
    # _build_calib_iter 也会返回 processor，此处保留 model 自己的 processor 以兼容性
    _ = dl_processor, processor  # quiet lint about unused

    scores = compute_first_order_proxy(
        model=model,
        dataloader=dl,
        device=device,
        bits=bits,
        max_batches=max_batches,
    )
    # 直接保存 mydesk 版输出（含 fo_score 与 proxy_{b}）
    cpu_scores = {k: {kk: vv.detach().float().cpu() for kk, vv in v.items()} for k, v in scores.items()}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(cpu_scores, out_path)
    print(f"[AutoQVLA][first_order] saved scores to {out_path}")


def build_bitwise_sensitivity_proxy(
    ckpt: str,
    first_order_path: str,
    out_path: str,
    bits: Tuple[int, ...],
    device: torch.device,
) -> None:
    """
    基于 mydesk 版一阶代理（已含 proxy_{b}）直接输出；若遇到旧格式（纯 fo_score），
    则回退到权重量化噪声近似。
    """
    first_scores = torch.load(first_order_path, map_location="cpu")
    model, _ = load_model_and_processor(ckpt, device)
    model.eval()

    results: Dict[str, Dict[str, torch.Tensor]] = {}

    for name, module in model.named_modules():
        if name not in first_scores:
            continue
        layer_entry = first_scores[name]
        layer_res: Dict[str, torch.Tensor] = {}

        if isinstance(layer_entry, dict) and any(k.startswith("proxy_") for k in layer_entry.keys()):
            # 新格式：直接读取 proxy_b
            for b in bits:
                key = f"proxy_{b}"
                if key in layer_entry:
                    v = layer_entry[key]
                    layer_res[f"mse_{b}"] = v.detach().float().cpu() if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32)
            # 若缺失某些 bit，可跳过
        else:
            # 旧格式：只有 fo_score，需要乘以权重量化噪声
            if not hasattr(module, "weight"):
                continue
            w = getattr(module, "weight", None)
            if w is None or not isinstance(w, torch.Tensor):
                continue
            score = layer_entry
            score = score.to(device=device, dtype=torch.float32) if torch.is_tensor(score) else torch.tensor(score, device=device, dtype=torch.float32)
            w_dev = w.detach().to(device=device, dtype=torch.float32)
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                w_flat = w_dev.view(w_dev.size(0), -1)  # [C_out, D]
            else:
                continue

            noise_dict = _quantization_row_noise_l2(w_flat, bits)
            for b, noise in noise_dict.items():
                if noise.numel() != score.numel():
                    n = min(noise.numel(), score.numel())
                    noise_use = noise[:n]
                    score_use = score[:n]
                else:
                    noise_use = noise
                    score_use = score
                sens = (noise_use * score_use).detach().cpu()
                layer_res[f"mse_{b}"] = sens

        if layer_res:
            results[name] = layer_res

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(results, out_path)
    print(f"[AutoQVLA][first_order] saved bitwise proxy sensitivity to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_checkpoint", type=str, required=True)
    p.add_argument("--calib_jsonl", type=str, required=True)
    p.add_argument("--first_order_out", type=str, required=True)
    p.add_argument("--cum_path", type=str, default="")
    p.add_argument(
        "--proxy_out",
        type=str,
        default="",
        help="若非空，则基于一阶分数 + 权重量化噪声，估计每个通道在比特 {0,2,4,8} 下的敏感度并保存 "
             "（输出格式与 compute_channel_errors.py 兼容，键为 mse_0/mse_2/...）。",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_samples", type=int, default=64)
    p.add_argument("--max_layers", type=int, default=None)
    p.add_argument("--bits", type=str, default="0,2,4,8", help="计算一阶代理时的候选位宽，用于 proxy_{b}")
    p.add_argument("--bit", type=int, default=4, help="使用哪一位宽的累积误差做对比（mse_BIT）")
    args = p.parse_args()

    device = torch.device(args.device)

    # 1) 先算一阶敏感度（bit 无关的通道代理分数）
    bits = tuple(int(x) for x in args.bits.split(",") if x.strip())

    save_first_order_scores(
        ckpt=args.pretrained_checkpoint,
        calib_jsonl=args.calib_jsonl,
        out_path=args.first_order_out,
        device=device,
        max_samples=args.max_samples,
        bits=bits,
        max_batches=None,  # 保持与 mydesk 实现一致；可按需扩展
    )

    # 2)（可选）基于一阶分数 + 权重量化噪声，构造按比特 {0,2,4,8} 的敏感度近似
    if args.proxy_out:
        build_bitwise_sensitivity_proxy(
            ckpt=args.pretrained_checkpoint,
            first_order_path=args.first_order_out,
            out_path=args.proxy_out,
            bits=bits,
            device=device,
        )

    # 3) 如果提供了累积指标路径，就顺便算 rank 相关并打印
    if args.cum_path:
        per_layer = compare_rank(args.first_order_out, args.cum_path, bit=args.bit)
        global_corr = per_layer.get("__global__", float("nan"))
        print(f"[AutoQVLA][rank] global Spearman (bit={args.bit}) ≈ {global_corr:.4f}")
        # 打印若干代表层
        items = [(k, v) for k, v in per_layer.items() if k != "__global__"]
        items = sorted(items, key=lambda kv: kv[1], reverse=True)
        print("[AutoQVLA][rank] top-10 layers by Spearman:")
        for name, c in items[:10]:
            print(f"  {name}: {c:.4f}")


if __name__ == "__main__":
    main()



