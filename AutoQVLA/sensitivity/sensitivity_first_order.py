"""
First-order (gradient-based) channel sensitivity scoring for AutoQVLA-style gating.

思想（对齐论文提到的“一阶近似”）：
- 我们把通道的量化误差视为对该层输出通道的一个小扰动 Δy_c。
- 对某个标量目标 T（如动作向量的 L2 范数）做一阶展开：
    ΔT ≈ (∂T/∂y_c) * Δy_c
- 因此，通道敏感度可以用 E[ |∂T/∂y_c * Δy_c| ] 或 E[ |y_c * ∂T/∂y_c| ] 来近似。

这里实现一个通用的一阶近似：
- 目标 T 取为 predict_action 输出的 L2 范数：T = ||action||_2^2 / 2
- 对每个 Linear/Conv2d 的输出通道 c，累积：
    score_c = mean_batch_pixel( |y_c * grad_y_c| )
"""

from typing import Dict, Iterable, Tuple, Optional

import torch
import torch.nn as nn

try:
    # 优先使用 tqdm 显示一阶敏感度计算进度；若不可用则退化为普通迭代
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):  # type: ignore
        return x


def _is_target_module(name: str, m: nn.Module, include_projector: bool) -> bool:
    # 永远跳过跨模态 projector 与动作头（action_head / lm_head）
    if name.startswith("projector.") or "action_head" in name or name.startswith("language_model.lm_head"):
        return False
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if name.startswith("language_model.") or name.startswith("vision_backbone."):
            return True
        # 即便 include_projector=True 也不量化 projector，保持用户要求
    return False


@torch.no_grad()
def _build_calib_iter(calib_jsonl: str, ckpt: str, device: torch.device, max_samples: int):
    """
    复用与 gatedquantv2 类似的 calib JSONL：每行 {\"text\": ..., \"image\": ...}
    用 HF Processor 构造 batch 输入。
    """
    from PIL import Image
    from transformers import AutoProcessor
    import json
    import os

    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

    class _Iter:
        def __iter__(self):
            cnt = 0
            with open(calib_jsonl, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "text" not in obj or "image" not in obj:
                        continue
                    img_path = obj["image"]
                    if not os.path.isfile(img_path):
                        continue
                    image = Image.open(img_path).convert("RGB")
                    text = obj.get("text", "")
                    if not isinstance(text, str):
                        text = str(text)
                    # 使用 batch 形式以兼容多模态 Processor 的签名
                    inputs = processor([text], [image], return_tensors="pt")
                    batch = {}
                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            batch[k] = v.to(device)
                        else:
                            batch[k] = v
                    yield batch
                    cnt += 1
                    if cnt >= max_samples:
                        break

    return processor, _Iter()


def _forward_for_logits(model: nn.Module, batch_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    为了一阶敏感度，避免使用 .generate() / predict_action（不可微），
    而是直接在 causal LM logits 空间定义标量目标：
        T = 1/2 * ||logits_last||^2.

    这样既满足“动作头输出的一阶近似”思想，又能端到端反传梯度。
    """
    kwargs: Dict[str, torch.Tensor] = {}
    for k in ("pixel_values", "input_ids", "attention_mask", "position_ids"):
        if k in batch_inputs:
            kwargs[k] = batch_inputs[k]
    # 保证可微：不使用 use_cache / generate
    kwargs.setdefault("use_cache", False)
    out = model(**kwargs, output_hidden_states=False, return_dict=True)
    logits = out.logits  # [B, L, V]
    # 取最后一个 token 的 logits 作为“动作相关”输出的代理
    return logits[:, -1, :]  # [B, V]


def compute_first_order_sensitivity(
    model: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    include_projector: bool = False,
    max_layers: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    对每个目标模块（Linear/Conv2d）的输出通道，计算一阶近似敏感度：
        score_c = E[ mean_{B,spatial} | y_c * grad_y_c | ].

    返回:
        { layer_name: Tensor[C_out] }  其中 C_out 为该层输出通道数。
    """
    model.eval()
    model.to(device)

    # 注册 forward + backward hook
    scores: Dict[str, torch.Tensor] = {}
    counts: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hooks(layer_name: str, module: nn.Module, kind: str):
        # kind: "linear" or "conv"

        def fwd_hook(m, inp, out):
            if not torch.is_tensor(out):
                return
            setattr(m, "_autoqvla_last_out", out.detach())

        def bwd_hook(m, grad_input, grad_output):
            if not hasattr(m, "_autoqvla_last_out"):
                return
            y = getattr(m, "_autoqvla_last_out")
            g = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
            if not (torch.is_tensor(y) and torch.is_tensor(g)):
                return
            # 只支持 Linear (B, *, C) 和 Conv2d (B, C, H, W)
            if kind == "linear":
                # reshape to [B*, C]
                y_flat = y.reshape(-1, y.shape[-1])
                g_flat = g.reshape(-1, g.shape[-1])
                prod = (y_flat * g_flat).abs()  # [B*, C]
                score_c = prod.mean(dim=0)  # [C]
            elif kind == "conv":
                if y.dim() < 2:
                    return
                # y,g: [B, C, H, W]
                prod = (y * g).abs()  # [B,C,H,W]
                score_c = prod.mean(dim=(0, 2, 3))  # [C]
            else:
                return

            if layer_name not in scores:
                scores[layer_name] = score_c.detach()
                counts[layer_name] = torch.ones(1, device=score_c.device)
            else:
                scores[layer_name] += score_c.detach()
                counts[layer_name] += 1

        return fwd_hook, bwd_hook

    target_layers: Dict[str, Tuple[nn.Module, str]] = {}
    for name, m in model.named_modules():
        if _is_target_module(name, m, include_projector):
            if isinstance(m, nn.Linear):
                kind = "linear"
            elif isinstance(m, nn.Conv2d):
                kind = "conv"
            else:
                continue
            target_layers[name] = (m, kind)

    layer_count = 0
    for name, (m, kind) in target_layers.items():
        fwd, bwd = make_hooks(name, m, kind)
        handles.append(m.register_forward_hook(fwd))
        handles.append(m.register_full_backward_hook(bwd))  # PyTorch >=1.10
        layer_count += 1
        if max_layers is not None and layer_count >= max_layers:
            break

    # 迭代数据：每个样本计算一次 logits & 梯度
    # 如 dataloader 实现了 __len__，tqdm 会显示完整进度；否则以流式计数显示。
    for batch in tqdm(dataloader, desc="[AutoQVLA] first-order sensitivity", dynamic_ncols=True):
        # 清理梯度
        model.zero_grad(set_to_none=True)
        # 前向：得到最后一个 token 的 logits 作为“动作近似输出”
        logits_last = _forward_for_logits(model, batch)  # [B, V]
        # 标量目标：1/2 * ||logits_last||^2
        loss = 0.5 * (logits_last.float() ** 2).mean()
        loss.backward()

    # 移除 hooks
    for h in handles:
        h.remove()

    # 平均化
    out: Dict[str, torch.Tensor] = {}
    for name, s in scores.items():
        c = counts[name].item()
        if c <= 0:
            continue
        out[name] = (s / c).detach().cpu()
    return out



