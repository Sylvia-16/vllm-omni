# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flux.2-klein transformer adapter backed by the local diffusers implementation.

The baseline worker constructs this class through vllm-omni's pipeline and
weight loader, so this module keeps the vllm-omni constructor and
``load_weights`` surface while delegating the model implementation to
``diffusers.models.transformers.transformer_flux2``.

This mirrors the Z-Image adapter: the previous in-tree reimplementation (fused
QKV / vLLM parallel-linear blocks) diverged numerically from diffusers (~6%
per-step on an apple-to-apple tensor dump), producing visibly different images.
Delegating compute to diffusers removes that divergence.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModel as DiffusersFlux2Transformer2DModel,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig

try:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
except Exception:

    def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)


logger = init_logger(__name__)


class Flux2Transformer2DModel(DiffusersFlux2Transformer2DModel):
    """vllm-omni-compatible wrapper around diffusers' Flux.2 transformer.

    Compute is fully delegated to the diffusers base class; this subclass only
    keeps the vllm-omni constructor surface (``od_config`` / ``quant_config``),
    the offload/HSDP block metadata used by the evictor and pipelined loader,
    and a diffusers-name ``load_weights``.
    """

    _repeated_blocks = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return (
            "transformer_blocks" in name or "single_transformer_blocks" in name
        ) and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
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
        od_config: "OmniDiffusionConfig | None" = None,
        quant_config: "QuantizationConfig | None" = None,
    ) -> None:
        if quant_config is not None:
            logger.warning("Ignoring quant_config for diffusers-backed Flux2Transformer2DModel.")

        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            timestep_guidance_channels=timestep_guidance_channels,
            mlp_ratio=mlp_ratio,
            axes_dims_rope=axes_dims_rope,
            rope_theta=rope_theta,
            eps=eps,
            guidance_embeds=guidance_embeds,
        )

        # NOTE: do NOT assign self.dtype — ModelMixin exposes `dtype` as a
        # read-only property derived from the parameters. Weights load under
        # set_default_torch_dtype(od_config.dtype), so it already returns the
        # correct dtype.

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name.removeprefix("transformer.")
            if lookup_name not in params_dict:
                raise KeyError(f"Unknown Flux.2 transformer weight: {original_name}")

            param = params_dict[lookup_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params


__all__ = ["Flux2Transformer2DModel"]
