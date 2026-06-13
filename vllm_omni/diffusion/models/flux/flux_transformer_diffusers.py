# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLUX transformer adapter backed by the local diffusers implementation.

The baseline worker still constructs this class through vllm-omni's pipeline
and weight loader, so this module keeps the vllm-omni constructor and
``load_weights`` surface while delegating the model implementation to
``diffusers.models.transformers.transformer_flux``.

This mirrors the flux2 / sd3 / z_image adapters added in commit "update
transformer" so that FLUX.1-dev runs the stock diffusers transformer (3
separate Q/K/V Linears, torch RMSNorm, diffusers attention processor) instead
of vllm-omni's fused-kernel reimplementation. Used to isolate how much of the
vllm-omni speedup comes from the custom transformer kernels.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel as DiffusersFluxTransformer2DModel,
)
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig

try:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
except Exception:

    def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)


logger = init_logger(__name__)


def _od_config_value(od_config: OmniDiffusionConfig | None, key: str, default: Any) -> Any:
    if od_config is None:
        return default
    return od_config.tf_model_config.get(key, default)


class FluxTransformer2DModel(DiffusersFluxTransformer2DModel):
    """vllm-omni-compatible wrapper around diffusers' FLUX transformer."""

    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]

    @staticmethod
    def _is_transformer_block(name: str, module: nn.Module) -> bool:
        return ("transformer_blocks" in name or "single_transformer_blocks" in name) and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        theta: float = 10000.0,
        quant_config: Any | None = None,
    ):
        if quant_config is not None:
            logger.warning("Ignoring quant_config for diffusers-backed FluxTransformer2DModel.")

        self.parallel_config = od_config.parallel_config if od_config is not None else None

        super().__init__(
            patch_size=_od_config_value(od_config, "patch_size", patch_size),
            in_channels=_od_config_value(od_config, "in_channels", in_channels),
            out_channels=_od_config_value(od_config, "out_channels", out_channels),
            num_layers=_od_config_value(od_config, "num_layers", num_layers),
            num_single_layers=_od_config_value(od_config, "num_single_layers", num_single_layers),
            attention_head_dim=_od_config_value(od_config, "attention_head_dim", attention_head_dim),
            num_attention_heads=_od_config_value(od_config, "num_attention_heads", num_attention_heads),
            joint_attention_dim=_od_config_value(od_config, "joint_attention_dim", joint_attention_dim),
            pooled_projection_dim=_od_config_value(od_config, "pooled_projection_dim", pooled_projection_dim),
            guidance_embeds=_od_config_value(od_config, "guidance_embeds", guidance_embeds),
            axes_dims_rope=_od_config_value(od_config, "axes_dims_rope", axes_dims_rope),
        )

        # vllm-omni's Flux pipeline reads these directly during latent /
        # guidance preparation (pipeline_flux.py: in_channels // 4,
        # transformer.guidance_embeds).
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels or self.config.in_channels
        self.guidance_embeds = self.config.guidance_embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        img_ids: torch.Tensor | None = None,
        txt_ids: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        # vllm-omni's CFG pipeline passes scalar conditioning inputs (timestep,
        # guidance) and the rope id tensors on CPU. Diffusers' forward only
        # casts dtype (transformer_flux.py: `timestep.to(hidden_states.dtype)`),
        # not device, so the first embedding Linear hits a cpu/cuda mismatch.
        # The fused-kernel transformer this adapter replaces tolerated CPU
        # inputs; restore that contract by coercing every tensor input onto the
        # model's device before delegating to diffusers.
        device = next(self.parameters()).device

        def _to_device(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if isinstance(tensor, torch.Tensor) and tensor.device != device:
                return tensor.to(device)
            return tensor

        return super().forward(
            hidden_states=_to_device(hidden_states),
            encoder_hidden_states=_to_device(encoder_hidden_states),
            pooled_projections=_to_device(pooled_projections),
            timestep=_to_device(timestep),
            img_ids=_to_device(img_ids),
            txt_ids=_to_device(txt_ids),
            guidance=_to_device(guidance),
            **kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name.removeprefix("transformer.")
            if lookup_name not in params_dict:
                raise KeyError(f"Unknown FLUX transformer weight: {original_name}")

            param = params_dict[lookup_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params


__all__ = [
    "FluxTransformer2DModel",
    "Transformer2DModelOutput",
]
