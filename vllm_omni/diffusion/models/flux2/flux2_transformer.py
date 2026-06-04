# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLUX 2 transformer adapter backed by the local diffusers implementation.

The baseline worker still constructs this class through vllm-omni's pipeline
and weight loader, so this module keeps the vllm-omni constructor and
``load_weights`` surface while delegating the model implementation to
``diffusers.models.transformers.transformer_flux2``.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2AttnProcessor,
    Flux2FeedForward,
    Flux2KVAttnProcessor,
    Flux2KVCache,
    Flux2Modulation,
    Flux2ParallelSelfAttention,
    Flux2ParallelSelfAttnProcessor,
    Flux2PosEmbed,
    Flux2SingleTransformerBlock,
    Flux2SwiGLU,
    Flux2TimestepGuidanceEmbeddings,
    Flux2Transformer2DModelOutput,
    Flux2TransformerBlock,
)
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModel as DiffusersFlux2Transformer2DModel,
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


class Flux2Transformer2DModel(DiffusersFlux2Transformer2DModel):
    """vllm-omni-compatible wrapper around diffusers' Flux2 transformer."""

    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]

    @staticmethod
    def _is_transformer_block(name: str, module: nn.Module) -> bool:
        return ("transformer_blocks" in name or "single_transformer_blocks" in name) and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
        quant_config: Any | None = None,
    ):
        if quant_config is not None:
            logger.warning("Ignoring quant_config for diffusers-backed Flux2Transformer2DModel.")

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
            timestep_guidance_channels=_od_config_value(od_config, "timestep_guidance_channels", timestep_guidance_channels),
            mlp_ratio=_od_config_value(od_config, "mlp_ratio", mlp_ratio),
            axes_dims_rope=_od_config_value(od_config, "axes_dims_rope", axes_dims_rope),
            rope_theta=_od_config_value(od_config, "rope_theta", rope_theta),
            eps=_od_config_value(od_config, "eps", eps),
            guidance_embeds=_od_config_value(od_config, "guidance_embeds", guidance_embeds),
        )

        # vllm-omni's Flux2 pipeline reads these values directly during
        # latent/guidance preparation.
        self.out_channels = self.config.out_channels or self.config.in_channels
        self.guidance_embeds = self.config.guidance_embeds

        # Keep vllm-omni's config hook so non-default RoPE theta still works.
        configured_theta = _od_config_value(od_config, "rope_theta", rope_theta)
        if configured_theta != 2000:
            self.pos_embed = Flux2PosEmbed(theta=configured_theta, axes_dim=self.config.axes_dims_rope)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name.removeprefix("transformer.")
            if lookup_name not in params_dict:
                raise KeyError(f"Unknown Flux2 transformer weight: {original_name}")

            param = params_dict[lookup_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params


__all__ = [
    "Flux2Attention",
    "Flux2AttnProcessor",
    "Flux2FeedForward",
    "Flux2KVAttnProcessor",
    "Flux2KVCache",
    "Flux2Modulation",
    "Flux2ParallelSelfAttention",
    "Flux2ParallelSelfAttnProcessor",
    "Flux2PosEmbed",
    "Flux2SingleTransformerBlock",
    "Flux2SwiGLU",
    "Flux2TimestepGuidanceEmbeddings",
    "Flux2Transformer2DModel",
    "Flux2Transformer2DModelOutput",
    "Flux2TransformerBlock",
]
