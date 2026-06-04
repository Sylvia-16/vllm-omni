# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SD3 transformer adapter backed by the local diffusers implementation.

The baseline worker still constructs this class through vllm-omni's pipeline
and weight loader, so this module keeps the vllm-omni constructor and
``load_weights`` surface while delegating the model implementation to
``diffusers.models.transformers.transformer_sd3``.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers.transformer_sd3 import (
    SD3Transformer2DModel as DiffusersSD3Transformer2DModel,
)
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig

try:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
except Exception:

    def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)


logger = init_logger(__name__)


class SD3Transformer2DModel(DiffusersSD3Transformer2DModel):
    """vllm-omni-compatible wrapper around diffusers' SD3 transformer."""

    _repeated_blocks = ["JointTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    @staticmethod
    def _is_transformer_block(name: str, module: nn.Module) -> bool:
        return "transformer_blocks" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
    ):
        model_config = od_config.tf_model_config

        self.parallel_config = od_config.parallel_config

        super().__init__(
            sample_size=model_config.sample_size,
            patch_size=model_config.patch_size,
            in_channels=model_config.in_channels,
            num_layers=model_config.num_layers,
            attention_head_dim=model_config.attention_head_dim,
            num_attention_heads=model_config.num_attention_heads,
            joint_attention_dim=model_config.joint_attention_dim,
            caption_projection_dim=model_config.caption_projection_dim,
            pooled_projection_dim=model_config.pooled_projection_dim,
            out_channels=model_config.out_channels,
            pos_embed_max_size=model_config.pos_embed_max_size,
            dual_attention_layers=getattr(model_config, "dual_attention_layers", ()),
            qk_norm=getattr(model_config, "qk_norm", None),
        )

        # vllm-omni pipeline reads these directly.
        self.num_layers = model_config.num_layers
        self.sample_size = model_config.sample_size
        self.in_channels = model_config.in_channels
        self.out_channels = model_config.out_channels
        self.num_attention_heads = model_config.num_attention_heads
        self.attention_head_dim = model_config.attention_head_dim
        self.inner_dim = model_config.num_attention_heads * model_config.attention_head_dim
        self.caption_projection_dim = model_config.caption_projection_dim
        self.pooled_projection_dim = model_config.pooled_projection_dim
        self.joint_attention_dim = model_config.joint_attention_dim
        self.patch_size = model_config.patch_size
        self.pos_embed_max_size = model_config.pos_embed_max_size

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())

        for name, buffer in self.named_buffers():
            if name.endswith(".pos_embed"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name.removeprefix("transformer.")
            if lookup_name not in params_dict:
                raise KeyError(f"Unknown SD3 transformer weight: {original_name}")

            param = params_dict[lookup_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params


__all__ = [
    "JointTransformerBlock",
    "SD3Transformer2DModel",
    "Transformer2DModelOutput",
]
