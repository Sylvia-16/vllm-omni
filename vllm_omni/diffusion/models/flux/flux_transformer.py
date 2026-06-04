# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLUX transformer adapter backed by the local diffusers implementation.

The baseline worker still constructs this class through vllm-omni's pipeline
and weight loader, so this module keeps the vllm-omni constructor and
``load_weights`` surface while delegating the model implementation to
``diffusers.models.transformers.transformer_flux``.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxAttnProcessor,
    FluxIPAdapterAttnProcessor,
    FluxPosEmbed,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel as DiffusersFluxTransformer2DModel,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

from vllm_omni.diffusion.data import OmniDiffusionConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

try:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
except Exception:

    def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)


logger = init_logger(__name__)


class ColumnParallelApproxGELU(nn.Module):
    """Keep the vLLM feed-forward primitive used by HunyuanVideo."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        approximate: str,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in,
            dim_out,
            bias=bias,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.approximate = approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate=self.approximate)


# class FeedForward(nn.Module):
#     """vLLM-parallel feed-forward retained for non-FLUX importers."""

#     def __init__(
#         self,
#         dim: int,
#         dim_out: int | None = None,
#         mult: int = 4,
#         activation_fn: str = "gelu-approximate",
#         inner_dim: int | None = None,
#         bias: bool = True,
#         quant_config: "QuantizationConfig | None" = None,
#         prefix: str = "",
#     ) -> None:
#         super().__init__()
#         if activation_fn != "gelu-approximate":
#             raise ValueError(f"Unsupported activation_fn for vLLM FeedForward: {activation_fn}")

#         inner_dim = inner_dim or int(dim * mult)
#         dim_out = dim_out or dim
#         self.net = nn.ModuleList(
#             [
#                 ColumnParallelApproxGELU(
#                     dim,
#                     inner_dim,
#                     approximate="tanh",
#                     bias=bias,
#                     quant_config=quant_config,
#                     prefix=f"{prefix}.net.0",
#                 ),
#                 nn.Identity(),
#                 RowParallelLinear(
#                     inner_dim,
#                     dim_out,
#                     input_is_parallel=True,
#                     return_bias=False,
#                     quant_config=quant_config,
#                     prefix=f"{prefix}.net.2",
#                 ),
#             ]
#         )

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         for module in self.net:
#             hidden_states = module(hidden_states)
#         return hidden_states


def _od_config_value(od_config: OmniDiffusionConfig | None, key: str, default: Any) -> Any:
    if od_config is None:
        return default
    return od_config.tf_model_config.get(key, default)


class FluxTransformer2DModel(DiffusersFluxTransformer2DModel):
    """vllm-omni-compatible wrapper around diffusers' Flux transformer."""

    # Preserve vllm-omni lifecycle metadata even though the block
    # implementation itself comes from diffusers.
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

        # vllm-omni's Flux pipeline predates ConfigMixin attribute fallback and
        # reads these values directly during latent/guidance preparation.
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels or self.config.in_channels
        self.guidance_embeds = self.config.guidance_embeds

        # Diffusers hard-codes FLUX.1's default theta. Keep vllm-omni's config
        # hook so non-default transformer configs still build the same RoPE.
        configured_theta = _od_config_value(od_config, "theta", theta)
        if configured_theta != 10000.0:
            self.pos_embed = FluxPosEmbed(theta=configured_theta, axes_dim=self.config.axes_dims_rope)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name.removeprefix("transformer.")
            if lookup_name not in params_dict:
                raise KeyError(f"Unknown Flux transformer weight: {original_name}")

            param = params_dict[lookup_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params


class FluxKontextTransformer2DModel(FluxTransformer2DModel):
    pass


__all__ = [
    "ColumnParallelApproxGELU",
    # "FeedForward",
    "FluxAttention",
    "FluxAttnProcessor",
    "FluxIPAdapterAttnProcessor",
    "FluxKontextTransformer2DModel",
    "FluxPosEmbed",
    "FluxSingleTransformerBlock",
    "FluxTransformer2DModel",
    "FluxTransformerBlock",
    "Transformer2DModelOutput",
]
