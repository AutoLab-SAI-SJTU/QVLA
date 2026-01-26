"""
Apply trained DuQuant parameters to an OpenVLA model by replacing LLM DecoderLayers at runtime.

This file is intentionally self-contained inside AutoQVLAv2 (no imports from external DuQuant paths).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from AutoQVLAv2.duquant.models.int_llama_layer import QuantLlamaDecoderLayer
from AutoQVLAv2.duquant.models.int_mistral_layer import QuantMistralDecoderLayer


@dataclass
class DuQuantRuntimeConfig:
    # Quant knobs (align names with DuQuant)
    wbits: int = 4
    abits: int = 16
    block_size: int = 128
    max_rotation_step: int = 256
    permutation_times: int = 1
    group_size: int = -1
    symmetric: bool = True
    no_rotate: bool = False


def _make_duquant_layer_args(cfg: DuQuantRuntimeConfig) -> SimpleNamespace:
    """
    Build the minimal `args` object required by DuQuant's quantized layer implementations.
    """
    gs = None if (cfg.group_size is None or int(cfg.group_size) <= 0) else int(cfg.group_size)

    weight_quant_params = dict(
        n_bits=int(cfg.wbits),
        symmetric=bool(cfg.symmetric),
        group_size=gs,
        dynamic_method="per_channel",
        quant_method="duquant",
        block_size=int(cfg.block_size),
        max_rotation_step=int(cfg.max_rotation_step),
        permutation_times=int(cfg.permutation_times),
    )
    act_quant_params = dict(
        n_bits=int(cfg.abits),
        symmetric=bool(cfg.symmetric),
        group_size=gs,
        dynamic_method="per_token",
        quant_method="duquant",
        block_size=int(cfg.block_size),
        max_rotation_step=int(cfg.max_rotation_step),
        permutation_times=int(cfg.permutation_times),
    )

    # DuQuant layers expect many per-projection fields; in the vanilla code they are typically identical.
    rotate = (not bool(cfg.no_rotate))
    return SimpleNamespace(
        wbits=int(cfg.wbits),
        abits=int(cfg.abits),
        rotate=rotate,
        disable_input_quant=False,

        # MLP projections
        gate_weight_quant_params=weight_quant_params,
        gate_act_quant_params=act_quant_params,
        up_weight_quant_params=weight_quant_params,
        up_act_quant_params=act_quant_params,
        down_weight_quant_params=weight_quant_params,
        down_act_quant_params=act_quant_params,

        # Attention projections
        q_weight_quant_params=weight_quant_params,
        q_act_quant_params=act_quant_params,
        k_weight_quant_params=weight_quant_params,
        k_act_quant_params=act_quant_params,
        v_weight_quant_params=weight_quant_params,
        v_act_quant_params=act_quant_params,
        o_weight_quant_params=weight_quant_params,
        o_act_quant_params=act_quant_params,

        # Attention matmul quant params (kept for init; forward may not use them)
        q_quant_params=act_quant_params,
        k_quant_params=act_quant_params,
        p_quant_params=act_quant_params,
        v_quant_params=act_quant_params,
    )


def _get_language_model(vla: nn.Module) -> nn.Module:
    if hasattr(vla, "language_model"):
        return getattr(vla, "language_model")
    if hasattr(vla, "model") and hasattr(vla.model, "language_model"):
        return getattr(vla.model, "language_model")
    raise AttributeError("Cannot find language model on VLA model (expected `.language_model` or `.model.language_model`).")


def _get_decoder_layers(lm: nn.Module):
    # HF LLaMA/Mistral convention: lm.model.layers
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    # Fallbacks
    if hasattr(lm, "layers"):
        return lm.layers
    raise AttributeError("Cannot find decoder layers on language model (expected `.model.layers` or `.layers`).")


def apply_trained_duquant_to_language_model(
    vla: nn.Module,
    duquant_parameters_path: str,
    *,
    runtime_cfg: Optional[DuQuantRuntimeConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, int]:
    """
    Replace each LLM DecoderLayer with DuQuant's Quant*DecoderLayer and load the trained parameters.

    Returns: (vla, num_layers_applied)
    """
    if runtime_cfg is None:
        runtime_cfg = DuQuantRuntimeConfig()
    if device is None:
        device = next(vla.parameters()).device

    params_by_layer: Any = torch.load(duquant_parameters_path, map_location="cpu")
    if not isinstance(params_by_layer, dict):
        raise ValueError(f"Unexpected duquant_parameters format: expected dict, got {type(params_by_layer)}")

    lm = _get_language_model(vla)
    layers = _get_decoder_layers(lm)

    args = _make_duquant_layer_args(runtime_cfg)

    model_type = getattr(getattr(lm, "config", None), "model_type", None)
    is_mistral = (model_type == "mistral") or ("mistral" in str(model_type).lower() if model_type else False)

    applied = 0
    for i in range(len(layers)):
        if i not in params_by_layer:
            continue

        ori_layer = layers[i]
        qlayer = QuantMistralDecoderLayer(lm.config, ori_layer, args) if is_mistral else QuantLlamaDecoderLayer(lm.config, ori_layer, args)
        # Keep dtype consistent with the original layer if possible
        ori_p = next(iter(ori_layer.parameters()), None)
        if ori_p is not None:
            qlayer.to(device=device, dtype=ori_p.dtype)
        else:
            qlayer.to(device=device)

        # Load per-layer DuQuant parameters (flat keys)
        sd = params_by_layer[i]
        qlayer.load_post_params(sd, device=device)
        qlayer.load_duquant_params(sd, device=device)
        qlayer.load_smooth_params(sd, device=device)
        qlayer.load_lwc_params(sd, device=device)
        # Make sure DuQuant rotation/permutation buffers are registered and init flags are set.
        if hasattr(qlayer, "register_duquant_params"):
            qlayer.register_duquant_params()

        # Enable quant in forward
        qlayer.set_quant_state(weight_quant=True, act_quant=True)

        layers[i] = qlayer
        applied += 1

    # Avoid transformers Cache object incompatibilities during eval
    if hasattr(vla, "config") and hasattr(vla.config, "use_cache"):
        vla.config.use_cache = False
    if hasattr(lm, "config") and hasattr(lm.config, "use_cache"):
        lm.config.use_cache = False

    return vla, applied


