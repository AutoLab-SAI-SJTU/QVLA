from typing import Dict, Optional

import os
import sys
import torch
import torch.nn as nn

from .kernels.triton_w4a4_gemm import (
    is_triton_available,
    pack_w4_rowwise,
    w4a4_mm,
    w4a16_mm,
)

try:
    # 复用 UniVLA gptq_gemm 的 INT8 Triton kernel（可选依赖）
    from gptq_gemm.kernels.triton_int8_gemm import int8_deq_mm  # type: ignore[import]
except Exception:  # pragma: no cover
    int8_deq_mm = None


def _load_quarot_components():
    """
    动态从 AutoQVLA/Activation/QuaRot 下加载 QuaRot 相关组件。

    返回:
        quarot, Quantizer, pack_i4, OnlineHadamard

    若导入失败，会给出清晰的报错提示，指导用户在本地编译/安装 QuaRot。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    quarot_root = os.path.join(here, "Activation", "QuaRot")
    if os.path.isdir(quarot_root) and quarot_root not in sys.path:
        sys.path.append(quarot_root)

    try:  # pragma: no cover - 依赖本地编译好的 QuaRot 扩展
        import quarot  # type: ignore[import]
        from quarot.nn.quantization import Quantizer  # type: ignore[import]
        from quarot.nn.hadamard import OnlineHadamard  # type: ignore[import]
        from quarot.functional.quantization import pack_i4  # type: ignore[import]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "AutoQVLA: 无法导入 QuaRot（需要真实 W4A4 kernel 时）。\n"
            f"请先在本地编译并安装 QuaRot：\n"
            f"  cd {quarot_root} && python -m pip install -e .\n"
            "随后重新导入 AutoQVLA 模块。"
        ) from e

    return quarot, Quantizer, pack_i4, OnlineHadamard


class AutoQVLALinearW4A4(nn.Module):
    """
    W4A4 linear layer with packed 4-bit weights and Triton GEMM.

    - Weight: per-output-channel symmetric 4-bit quantization, packed row-wise (2 weights / int8).
    - Activation: per-input-row 4-bit quantization on-the-fly inside the kernel (no large buffers).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # placeholders; real quantization is done offline then loaded via load_state_dict
        self.register_buffer("qweight_packed", torch.empty(0, dtype=torch.int8))
        self.register_buffer("w_scales", torch.ones((out_features, 1), dtype=torch.float16))
        self.register_buffer("a_scales", torch.ones(1, dtype=torch.float16))  # scalar or [1]
        self._k_orig: int = in_features

        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16)) if bias else None

        # meta info for pack/unpack; kept for compatibility / debugging
        self.pack_meta: Dict[str, int] = {"k_orig": in_features}

    @torch.no_grad()
    def quantize_from_float(
        self,
        weight: torch.Tensor,
        w_scales: Optional[torch.Tensor] = None,
        num_bits: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Quantize float16/bfloat16 weights to 4-bit containers and pack (mixed-precision aware).

        设计要点（结合 W4A4 混合精度场景）：
        - 物理存储始终使用 4bit 容器（2 参数 / byte，便于高效打包与访存）；
        - 逻辑 bit 数由 num_bits 控制，当前支持每个输出通道独立为 2bit 或 4bit；
          * 2bit 通道：qmax=1，scale=max_abs/1，有效量化级别 ∈{-1,0,1}；
          * 4bit 通道：qmax=7，scale=max_abs/7，有效量化级别 ∈[-7,...,7}；
        - kernel 侧统一按「int4 + per-channel scale」进行反量化与 GEMM 计算，
          混合精度完全通过不同通道的 scale / 有效量化级体现，不引入额外分支或内存。

        Args:
            weight: [out_features, in_features] float tensor.
            w_scales: 可选的行级 scale，[out_features, 1]；若为 None，则按 num_bits 自适应计算。
            num_bits: 可选的每通道 bit 数，[out_features]，元素取自 {2,4}。
        """
        assert weight.shape == (self.out_features, self.in_features)
        w = weight.detach().to(torch.float16)

        if w_scales is None:
            # per-row symmetric scale; 结合 num_bits 支持 2/4bit 混合
            max_abs = w.abs().amax(dim=1, keepdim=True)
            max_abs = torch.clamp(max_abs, min=1e-6)  # avoid div by zero

            if num_bits is None:
                # 统一 4bit：qmax=7
                qmax = torch.full_like(max_abs, 7.0, dtype=torch.float16)
            else:
                if num_bits.dim() != 1 or num_bits.numel() != self.out_features:
                    raise ValueError("num_bits 形状必须为 [out_features]")
                if not torch.all((num_bits == 2) | (num_bits == 4)):
                    raise ValueError("AutoQVLALinearW4A4 目前仅支持 2/4 bit 通道混合")
                # [out,1]，2bit->1, 4bit->7
                num_bits_row = num_bits.view(-1, 1).to(dtype=torch.int32)
                qmax = ((1 << (num_bits_row - 1)) - 1).to(dtype=torch.float16)

            w_scales = max_abs / qmax

        qmax_store = 7  # 物理容器仍然是 int4 范围 [-8,7]
        q = torch.round(w / w_scales).clamp_(-qmax_store - 1, qmax_store).to(torch.int8)

        packed, k_orig = pack_w4_rowwise(q)
        self.qweight_packed = packed
        self.w_scales = w_scales.to(dtype=torch.float16, device=packed.device)
        self._k_orig = int(k_orig)
        self.pack_meta["k_orig"] = int(k_orig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        k = orig_shape[-1]
        if k != self.in_features:
            raise RuntimeError(f"AutoQVLALinearW4A4: expected last dim {self.in_features}, got {k}")
        x2 = x.reshape(-1, k)

        if self.qweight_packed.numel() > 0 and is_triton_available() and x2.is_cuda:
            # build per-row activation scales: simple max-abs / 7
            with torch.no_grad():
                a = x2.to(torch.float16)
                max_abs = a.abs().amax(dim=1, keepdim=True).clamp_(min=1e-6)
                a_scales = (max_abs / 7.0).to(torch.float16)

            y2 = w4a4_mm(
                a,
                self.qweight_packed,
                self.w_scales,
                a_scales,
                K_orig=self._k_orig,
                out_dtype=a.dtype,
            )
            if y2 is None:
                raise RuntimeError("AutoQVLALinearW4A4: Triton W4A4 kernel unavailable on CUDA")
            if self.bias is not None:
                y2 = y2 + self.bias.to(dtype=y2.dtype, device=y2.device)
            return y2.reshape(*orig_shape[:-1], self.out_features)

        # CPU / fallback: dequantize weights and activations on the fly and matmul in float
        # Note: this path is not optimized and mainly for correctness.
        if self.qweight_packed.numel() == 0:
            raise RuntimeError("AutoQVLALinearW4A4: qweight_packed is empty")
        # unpack weights using pack_w4_rowwise inverse
        packed = self.qweight_packed.to(torch.int8)
        N, Kp = packed.shape
        K = self._k_orig
        # unpack to [N, 2*Kp], then slice K
        u = packed.to(torch.uint8)
        low = (u & 0xF).to(torch.int8)
        high = ((u >> 4) & 0xF).to(torch.int8)
        full = torch.empty((N, Kp * 2), dtype=torch.int8, device=packed.device)
        full[:, 0::2] = low
        full[:, 1::2] = high
        full = full[:, :K]
        w_int4 = torch.where(full > 7, full - 16, full).to(torch.float32)
        w = (w_int4 * self.w_scales.to(dtype=torch.float32)).to(dtype=x2.dtype)
        # simple activation quant/dequant
        a = x2.to(torch.float32)
        max_abs = a.abs().amax(dim=1, keepdim=True).clamp_(min=1e-6)
        a_scales = max_abs / 7.0
        a_q = torch.round(a / a_scales).clamp_(-8, 7)
        a_deq = (a_q * a_scales).to(dtype=x2.dtype)
        y2 = a_deq.matmul(w.t())
        if self.bias is not None:
            y2 = y2 + self.bias.to(dtype=y2.dtype, device=y2.device)
        return y2.reshape(*orig_shape[:-1], self.out_features)


class AutoQVLALinearW4A4QuaRot(nn.Module):
    """
    基于 QuaRot INT4 GEMM 的 W4A4 线性层（真实 int4 kernel）：

    - 权重：每个输出通道 4bit，对称量化 + 行级 scale，使用 QuaRot 的 pack_i4 打包；
    - 激活：使用 QuaRot 的 Quantizer 做对称 4bit 量化，输出 PackedQuantizedTensor；
    - 前向：调用 quarot.matmul 做 INT4xINT4->INT32 矩阵乘法，再用 quarot.sym_dequant 反量化得到 fp16/bf16 输出。

    注意：
    - 依赖本地已编译好的 QuaRot 扩展（quarot._CUDA），若未安装会抛出 ImportError 并提示安装命令；
    - 要求 in_features 是 32 的倍数（QuaRot kernel 限制）。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_clip_ratio: float = 1.0,
        use_hadamard: bool = False,
        hadamard_force_fp32: bool = False,
    ):
        super().__init__()
        if in_features % 32 != 0:
            raise ValueError(f"AutoQVLALinearW4A4QuaRot: in_features 必须是 32 的倍数, got {in_features}")

        self.in_features = in_features
        self.out_features = out_features
        self.input_clip_ratio = float(input_clip_ratio)

        # 权重：打包后的 int4（2 权重 / byte），行级 scale
        self.register_buffer("qweight", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("w_scales", torch.ones((out_features, 1), dtype=torch.float16))

        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16)) if bias else None

        # 延迟加载 QuaRot 组件（避免在未安装时导入失败）
        quarot, Quantizer, _, OnlineHadamard = _load_quarot_components()
        self._quarot = quarot
        # 激活量化器：与 QuaRot 自带 Quantizer 一致（对称 4bit，行级动态 scale）
        self.act_quantizer = Quantizer(input_clip_ratio=self.input_clip_ratio)
        # 可选：在激活侧启用 Hadamard 旋转（DuQuant 风格旋转接口占位）
        if use_hadamard:
            try:
                self.hadamard = OnlineHadamard(in_features, force_fp32=hadamard_force_fp32)
            except Exception:
                # 若当前 hidden dim 无法构造 Hadamard，则退化为禁用
                self.hadamard = None
        else:
            self.hadamard = None

    @torch.no_grad()
    def quantize_from_float(self, weight: torch.Tensor, w_scales: Optional[torch.Tensor] = None) -> None:
        """
        离线权重量化 + 打包（DuQuant 风格中的“权重量化 + 真实 kernel”部分）：

        - 输入全精度权重 [out, in]；
        - 按行做 max-abs / 7 的对称 4bit 量化（或使用给定的 w_scales）；
        - 使用 QuaRot 的 pack_i4 打包成 [out, in/2] 的 uint8（2 个权重 / byte）。
        """
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"AutoQVLALinearW4A4QuaRot.quantize_from_float: "
                f"expect weight shape {(self.out_features, self.in_features)}, got {tuple(weight.shape)}"
            )

        quarot, _, pack_i4, _ = _load_quarot_components()

        w = weight.detach().to(torch.float16)
        if w_scales is None:
            max_abs = w.abs().amax(dim=1, keepdim=True).clamp_(min=1e-6)
            w_scales = max_abs / 7.0  # 对称 4bit，范围 [-7,7]

        # 整数权重（仍在 [-8,7] 范围存储）
        int_rounded = torch.round(w / w_scales).clamp_(-8, 7).to(torch.int8)
        # QuaRot pack_i4: [out, in] (int8, signed) -> [out, in/2] (uint8)
        q_packed = pack_i4(int_rounded)

        if not q_packed.is_contiguous():
            q_packed = q_packed.contiguous()

        # 保存到当前设备
        device = w.device
        self.qweight = q_packed.to(device=device, dtype=torch.uint8)
        self.w_scales = w_scales.to(dtype=torch.float16, device=device)

        # 记录一下 QuaRot 句柄，forward 会重复使用
        self._quarot = quarot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向：
        - 使用 QuaRot 的 Quantizer 对激活做 4bit 量化（得到 PackedQuantizedTensor）；
        - 调用 quarot.matmul 做 int4 GEMM；
        - 用 quarot.sym_dequant 结合激活/权重的 scale 反量化为 fp16/bf16；
        - 可选加上偏置并恢复原始 batch 形状。
        """
        if self.qweight.numel() == 0:
            raise RuntimeError(
                "AutoQVLALinearW4A4QuaRot: qweight 为空，请先调用 quantize_from_float 进行权重量化。"
            )

        if not x.is_cuda:
            raise RuntimeError("AutoQVLALinearW4A4QuaRot 目前只支持 CUDA 张量（依赖 quarot._CUDA kernel）。")

        quarot = self._quarot
        if quarot is None:
            quarot, _, _, _ = _load_quarot_components()
            self._quarot = quarot

        orig_shape = x.shape
        k = orig_shape[-1]
        if k != self.in_features:
            raise RuntimeError(f"AutoQVLALinearW4A4QuaRot: expected last dim {self.in_features}, got {k}")

        # QuaRot 的 Quantizer 按最后一维做行级 max-abs scale
        x2 = x.reshape(-1, k).to(torch.float16)
        # 可选：在激活侧应用 Hadamard 旋转（占位 DuQuant 风格旋转接口）
        if getattr(self, "hadamard", None) is not None:
            x2 = self.hadamard(x2)
        packed_x = self.act_quantizer(x2)  # PackedQuantizedTensor
        qx = packed_x.quantized_x  # [M, in/2], uint8（打包后的 int4）
        sx = packed_x.scales_x  # [M, 1], fp16

        # INT4 GEMM： [M, K/2] x [out, K/2] -> [M, out] (int32)
        qy = quarot.matmul(qx, self.qweight)

        # 反量化：y = qy * sx * w_scales
        y2 = quarot.sym_dequant(qy, sx, self.w_scales)  # [M, out]
        if self.bias is not None:
            y2 = y2 + self.bias.to(dtype=y2.dtype, device=y2.device)

        return y2.reshape(*orig_shape[:-1], self.out_features)


class AutoQVLALinearW4A16(nn.Module):
    """
    纯权重量化版本：W4A16

    - 权重：每个输出通道 4bit，对称量化 + 行级 scale，使用 pack_w4_rowwise 打包；
    - 激活：保持 fp16/bf16，不做 DuQuant 激活量化；
    - 前向：调用 Triton w4a16_mm 内核在打包权重上做 GEMM。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("qweight_packed", torch.empty(0, dtype=torch.int8))
        self.register_buffer("w_scales", torch.ones((out_features, 1), dtype=torch.float16))
        self._k_orig: int = in_features

        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16)) if bias else None
        self.pack_meta: Dict[str, int] = {"k_orig": in_features}

    @torch.no_grad()
    def quantize_from_float(
        self,
        weight: torch.Tensor,
        w_scales: Optional[torch.Tensor] = None,
        num_bits: int = 4,
    ) -> None:
        """
        离线权重量化 + 打包：
        - 输入全精度权重 [out, in]；
        - num_bits 支持 2 或 4，比特数影响实际有效量化级，但仍用 4bit 存储；
        - 按行做 max-abs / qmax 的对称量化；
        - 用 pack_w4_rowwise 打成 [out, ceil(in/2)] 的 int8（2 个权重 / byte）。
        """
        assert weight.shape == (self.out_features, self.in_features)
        assert num_bits in (2, 4), f"AutoQVLALinearW4A16 目前只支持 2/4 bit, got {num_bits}"
        w = weight.detach().to(torch.float16)
        if w_scales is None:
            max_abs = w.abs().amax(dim=1, keepdim=True).clamp_(min=1e-6)
            qmax = (1 << (num_bits - 1)) - 1  # 2bit->1, 4bit->7
            w_scales = max_abs / float(qmax)
        qmax_store = 7  # 物理存储仍然使用 int4 范围 [-8,7]
        q = torch.round(w / w_scales).clamp_(-qmax_store - 1, qmax_store).to(torch.int8)
        packed, k_orig = pack_w4_rowwise(q)
        self.qweight_packed = packed
        self.w_scales = w_scales.to(dtype=torch.float16, device=packed.device)
        self._k_orig = int(k_orig)
        self.pack_meta["k_orig"] = int(k_orig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向：
        - 仅使用 4bit 权重 + scale，激活保持原精度；
        - CUDA 上优先走 Triton w4a16_mm 打包内核；
        - 其它情况 fallback 到 Python 解包 + matmul。
        """
        orig_shape = x.shape
        k = orig_shape[-1]
        if k != self.in_features:
            raise RuntimeError(f"AutoQVLALinearW4A16: expected last dim {self.in_features}, got {k}")
        x2 = x.reshape(-1, k)

        if self.qweight_packed.numel() > 0 and is_triton_available() and x2.is_cuda:
            a = x2.to(torch.float16)
            y2 = w4a16_mm(
                a,
                self.qweight_packed,
                self.w_scales,
                K_orig=self._k_orig,
                out_dtype=a.dtype,
            )
            if y2 is None:
                raise RuntimeError("AutoQVLALinearW4A16: Triton W4A16 kernel unavailable on CUDA")
            if self.bias is not None:
                y2 = y2 + self.bias.to(dtype=y2.dtype, device=y2.device)
            return y2.reshape(*orig_shape[:-1], self.out_features)

        # Fallback：CPU 或无 Triton 时，Python 解包 + matmul
        if self.qweight_packed.numel() == 0:
            raise RuntimeError("AutoQVLALinearW4A16: qweight_packed is empty")
        packed = self.qweight_packed.to(torch.int8)
        N, Kp = packed.shape
        K = self._k_orig
        u = packed.to(torch.uint8)
        low = (u & 0xF).to(torch.int8)
        high = ((u >> 4) & 0xF).to(torch.int8)
        full = torch.empty((N, Kp * 2), dtype=torch.int8, device=packed.device)
        full[:, 0::2] = low
        full[:, 1::2] = high
        full = full[:, :K]
        w_int4 = torch.where(full > 7, full - 16, full).to(torch.float32)
        w = (w_int4 * self.w_scales.to(dtype=torch.float32)).to(dtype=x2.dtype)
        y2 = x2.matmul(w.t())
        if self.bias is not None:
            y2 = y2 + self.bias.to(dtype=y2.dtype, device=y2.device)
        return y2.reshape(*orig_shape[:-1], self.out_features)


class AutoQVLALinearW8A16(nn.Module):
    """
    纯权重量化版本：W8A16

    - 权重：每个输出通道 8bit，对称量化 + 行级 scale，直接存 int8；
    - 激活：保持 fp16/bf16，不做激活量化；
    - 前向：当前实现为解量化 + matmul（后续可无缝替换为 GPTQ INT8 GEMM 内核）。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("qweight", torch.empty(0, dtype=torch.int8))
        self.register_buffer("w_scales", torch.ones((out_features, 1), dtype=torch.float16))

        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16)) if bias else None

    @torch.no_grad()
    def quantize_from_float(self, weight: torch.Tensor, w_scales: Optional[torch.Tensor] = None) -> None:
        """
        离线权重量化：
        - 输入全精度权重 [out, in]；
        - 按行做 max-abs / 127 的对称 8bit 量化；
        - 存为 int8 + 行级 scale。
        """
        assert weight.shape == (self.out_features, self.in_features)
        w = weight.detach().to(torch.float16)
        if w_scales is None:
            max_abs = w.abs().amax(dim=1, keepdim=True).clamp_(min=1e-6)
            w_scales = max_abs / 127.0
        q = torch.round(w / w_scales).clamp_(-128, 127).to(torch.int8)
        self.qweight = q
        self.w_scales = w_scales.to(dtype=torch.float16, device=q.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向：
        - 使用 8bit 权重 + scale，激活保持原精度；
        - CUDA 上若可用 gptq_gemm 的 Triton INT8 kernel，则优先调用；
        - 否则退化为解量化 + matmul 参考路径。
        """
        if self.qweight.numel() == 0:
            raise RuntimeError("AutoQVLALinearW8A16: qweight is empty (did you call quantize_from_float?)")

        orig_shape = x.shape
        k = orig_shape[-1]
        if k != self.in_features:
            raise RuntimeError(f"AutoQVLALinearW8A16: expected last dim {self.in_features}, got {k}")
        x2 = x.reshape(-1, k)

        # 优先使用 Triton INT8 GEMM 内核（若可用）
        if int8_deq_mm is not None and x2.is_cuda:
            y2 = int8_deq_mm(
                x2.to(dtype=torch.bfloat16 if x2.dtype == torch.bfloat16 else torch.float16),
                self.qweight,
                self.w_scales,
                out_dtype=x2.dtype if x2.dtype in (torch.float16, torch.bfloat16) else torch.float16,
            )
            if y2 is not None:
                if self.bias is not None:
                    y2 = y2 + self.bias.to(dtype=y2.dtype, device=y2.device)
                return y2.reshape(*orig_shape[:-1], self.out_features)

        # CPU 或无 Triton/gptq_gemm 时：解量化 + matmul 参考实现
        w = (self.qweight.float() * self.w_scales.float()).to(dtype=x2.dtype, device=x2.device)
        y2 = x2.matmul(w.t())
        if self.bias is not None:
            y2 = y2 + self.bias.to(dtype=y2.dtype, device=y2.device)
        return y2.reshape(*orig_shape[:-1], self.out_features)


class AutoQVLALinearMixedW(nn.Module):
    """
    通道级混合 bit 纯权重量化线性层：

    - 输入：与原始 Linear 相同；
    - 输出：按 gate 将通道划分为 {16, 8, 4, 2, 0} 五类：
        * 16bit: 使用一个子 Linear（全精度）；
        * 8bit: 使用 AutoQVLALinearW8A16；
        * 4bit: 使用 AutoQVLALinearW4A16(num_bits=4)；
        * 2bit: 使用 AutoQVLALinearW4A16(num_bits=2)，存储仍为 4bit；
        * 0bit: 通道剪枝，始终输出 0。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        idx_16: Optional[torch.Tensor],
        idx_8: Optional[torch.Tensor],
        idx_4: Optional[torch.Tensor],
        idx_2: Optional[torch.Tensor],
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 保存各 bit 组的索引，用 buffer 方便随模块一起搬设备
        self.register_buffer("idx_16", idx_16 if idx_16 is not None else torch.empty(0, dtype=torch.long))
        self.register_buffer("idx_8", idx_8 if idx_8 is not None else torch.empty(0, dtype=torch.long))
        self.register_buffer("idx_4", idx_4 if idx_4 is not None else torch.empty(0, dtype=torch.long))
        self.register_buffer("idx_2", idx_2 if idx_2 is not None else torch.empty(0, dtype=torch.long))

        self.linear_16 = (
            nn.Linear(in_features, int(self.idx_16.numel()), bias=bias) if self.idx_16.numel() > 0 else None
        )
        self.linear_8 = (
            AutoQVLALinearW8A16(in_features, int(self.idx_8.numel()), bias=bias)
            if self.idx_8.numel() > 0
            else None
        )
        self.linear_4 = (
            AutoQVLALinearW4A16(in_features, int(self.idx_4.numel()), bias=bias)
            if self.idx_4.numel() > 0
            else None
        )
        self.linear_2 = (
            AutoQVLALinearW4A16(in_features, int(self.idx_2.numel()), bias=bias)
            if self.idx_2.numel() > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        k = orig_shape[-1]
        if k != self.in_features:
            raise RuntimeError(f"AutoQVLALinearMixedW: expected last dim {self.in_features}, got {k}")

        # 初始化输出为 0（0bit 通道自然为 0）
        y = x.new_zeros(*orig_shape[:-1], self.out_features)

        if self.linear_16 is not None and self.idx_16.numel() > 0:
            y_16 = self.linear_16(x)
            y[..., self.idx_16] = y_16

        if self.linear_8 is not None and self.idx_8.numel() > 0:
            y_8 = self.linear_8(x)
            y[..., self.idx_8] = y_8

        if self.linear_4 is not None and self.idx_4.numel() > 0:
            y_4 = self.linear_4(x)
            y[..., self.idx_4] = y_4

        if self.linear_2 is not None and self.idx_2.numel() > 0:
            y_2 = self.linear_2(x)
            y[..., self.idx_2] = y_2

        return y


@torch.no_grad()
def build_mixedw_from_linear_and_gate(
    linear: nn.Linear,
    gate_bits: torch.Tensor,
) -> AutoQVLALinearMixedW:
    """
    从一个全精度 Linear + 通道 gate（0/2/4/8/16bit）构造 AutoQVLALinearMixedW。

    Args:
        linear: 原始 nn.Linear（权重形状 [out, in]）。
        gate_bits: 长度为 out 的整数张量，元素取自 {0,2,4,8,16}。
    """
    if gate_bits.dim() != 1 or gate_bits.numel() != linear.out_features:
        raise ValueError("gate_bits 形状必须为 [out_features]")

    gate_bits = gate_bits.to(torch.int64)
    in_f = linear.in_features
    out_f = linear.out_features

    idx_16 = (gate_bits == 16).nonzero(as_tuple=True)[0]
    idx_8 = (gate_bits == 8).nonzero(as_tuple=True)[0]
    idx_4 = (gate_bits == 4).nonzero(as_tuple=True)[0]
    idx_2 = (gate_bits == 2).nonzero(as_tuple=True)[0]
    # 0bit 通道自动被剪枝，不需要索引

    mixed = AutoQVLALinearMixedW(
        in_features=in_f,
        out_features=out_f,
        idx_16=idx_16,
        idx_8=idx_8,
        idx_4=idx_4,
        idx_2=idx_2,
        bias=linear.bias is not None,
    )

    device = linear.weight.device
    dtype = linear.weight.dtype
    mixed.to(device=device, dtype=dtype)

    # 16bit 子层：直接拷贝权重/偏置
    if mixed.linear_16 is not None and idx_16.numel() > 0:
        with torch.no_grad():
            mixed.linear_16.weight.copy_(linear.weight[idx_16].to(dtype=dtype, device=device))
            if linear.bias is not None and mixed.linear_16.bias is not None:
                mixed.linear_16.bias.copy_(linear.bias[idx_16].to(dtype=dtype, device=device))

    # 8bit 子层
    if mixed.linear_8 is not None and idx_8.numel() > 0:
        w8 = linear.weight[idx_8].to(dtype=dtype, device=device)
        mixed.linear_8.quantize_from_float(w8)
        if linear.bias is not None and mixed.linear_8.bias is not None:
            mixed.linear_8.bias.data.copy_(linear.bias[idx_8].to(dtype=torch.float16, device=device))

    # 4bit 子层
    if mixed.linear_4 is not None and idx_4.numel() > 0:
        w4 = linear.weight[idx_4].to(dtype=dtype, device=device)
        mixed.linear_4.quantize_from_float(w4, num_bits=4)
        if linear.bias is not None and mixed.linear_4.bias is not None:
            mixed.linear_4.bias.data.copy_(linear.bias[idx_4].to(dtype=torch.float16, device=device))

    # 2bit 子层（存为 4bit，但量化级为 2bit）
    if mixed.linear_2 is not None and idx_2.numel() > 0:
        w2 = linear.weight[idx_2].to(dtype=dtype, device=device)
        mixed.linear_2.quantize_from_float(w2, num_bits=2)
        if linear.bias is not None and mixed.linear_2.bias is not None:
            mixed.linear_2.bias.data.copy_(linear.bias[idx_2].to(dtype=torch.float16, device=device))

    return mixed


