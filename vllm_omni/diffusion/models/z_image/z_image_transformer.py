# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# _sp_plan definition adapted from HuggingFace diffusers library (_cp_plan)

# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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
"""Z-Image transformer adapter backed by the local diffusers implementation.

The baseline worker still constructs this class through vllm-omni's pipeline
and weight loader, so this module keeps the vllm-omni constructor and
``load_weights`` surface while delegating the model implementation to
``diffusers.models.transformers.transformer_z_image``.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_z_image import (
    FeedForward,
    FinalLayer,
    RopeEmbedder,
    TimestepEmbedder,
    ZImageTransformerBlock,
    ZSingleStreamAttnProcessor,
)
from diffusers.models.transformers.transformer_z_image import (
    ZImageTransformer2DModel as DiffusersZImageTransformer2DModel,
)
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CachedTransformer
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

try:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
except Exception:

    def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)


logger = init_logger(__name__)


def _positive_divisors(n: int) -> set[int]:
    if n <= 0:
        return set()
    divs: set[int] = set()
    import math as _math

    for d in range(1, int(_math.isqrt(n)) + 1):
        if n % d == 0:
            divs.add(d)
            divs.add(n // d)
    return divs


def validate_zimage_tp_constraints(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    in_channels: int,
    all_patch_size: tuple[int, ...],
    all_f_patch_size: tuple[int, ...],
    tensor_parallel_size: int,
) -> tuple[int, list[int], list[int]]:
    """Validate Z-Image TP constraints without requiring a distributed context.

    Returns:
        (ffn_hidden_dim, final_out_dims, supported_tp_candidates)
    """
    import math as _math

    tp_size = int(tensor_parallel_size)
    if tp_size <= 0:
        raise ValueError(f"tensor_parallel_size must be > 0, got {tp_size}")
    if dim % n_heads != 0:
        raise ValueError(f"dim must be divisible by n_heads, got dim={dim}, n_heads={n_heads}")
    if dim % tp_size != 0:
        supported = sorted(_positive_divisors(dim))
        raise ValueError(
            f"Z-Image requires dim % tensor_parallel_size == 0, but got dim={dim}, tp={tp_size}. "
            f"Supported tp candidates by dim: {supported}"
        )
    if n_heads % tp_size != 0:
        supported = sorted(_positive_divisors(n_heads))
        raise ValueError(
            f"Z-Image requires n_heads % tensor_parallel_size == 0, but got n_heads={n_heads}, tp={tp_size}. "
            f"Supported tp candidates by n_heads: {supported}"
        )
    if n_kv_heads % tp_size != 0:
        supported = sorted(_positive_divisors(n_kv_heads))
        raise ValueError(
            f"Z-Image requires n_kv_heads % tensor_parallel_size == 0, but got n_kv_heads={n_kv_heads}, "
            f"tp={tp_size}. Supported tp candidates by n_kv_heads: {supported}"
        )

    ffn_hidden_dim = int(dim / 3 * 8)
    if ffn_hidden_dim % tp_size != 0:
        supported = sorted(_positive_divisors(ffn_hidden_dim))
        raise ValueError(
            "Z-Image requires ffn_hidden_dim % tensor_parallel_size == 0 (for TP-sharded MLP), but got "
            f"ffn_hidden_dim={ffn_hidden_dim}, tp={tp_size}. Supported tp candidates by ffn_hidden_dim: {supported}"
        )

    final_out_dims = [
        int(patch_size) * int(patch_size) * int(f_patch_size) * int(in_channels)
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size)
    ]
    bad_final_out_dims = [d for d in final_out_dims if d % tp_size != 0]
    if bad_final_out_dims:
        supported = sorted(_positive_divisors(_math.gcd(*final_out_dims)))
        raise ValueError(
            "Z-Image requires final projection out_features divisible by tensor_parallel_size, but got "
            f"final_out_dims={final_out_dims}, tp={tp_size}. "
            f"Supported tp candidates by final_out_dims gcd: {supported}"
        )

    supported_tp_candidates = sorted(
        _positive_divisors(n_heads)
        & _positive_divisors(n_kv_heads)
        & _positive_divisors(dim)
        & _positive_divisors(ffn_hidden_dim)
        & _positive_divisors(_math.gcd(*final_out_dims))
    )
    return ffn_hidden_dim, final_out_dims, supported_tp_candidates


class ZImageTransformer2DModel(DiffusersZImageTransformer2DModel):
    """vllm-omni-compatible wrapper around diffusers' Z-Image transformer.

    Sequence Parallelism:
        This model supports non-intrusive SP via _sp_plan. The plan specifies:
        - Input splitting at first main transformer block (unified sequence)
        - RoPE (cos/sin) splitting along sequence dimension
        - Attention mask splitting along sequence dimension
        - Output gathering at final_layer

        Note: Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP) in diffusers.
    """

    _repeated_blocks = ["ZImageTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["layers"]

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return "layers" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    # Sequence Parallelism for Z-Image (following diffusers' _cp_plan pattern)
    _sp_plan = {
        "unified_prepare": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),
            1: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),
            2: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),
            3: SequenceParallelInput(split_dim=1, expected_dims=2, split_output=True),
        },
        "all_final_layer.2-1": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        quant_config: "QuantizationConfig | None" = None,
    ) -> None:
        if quant_config is not None:
            logger.warning("Ignoring quant_config for diffusers-backed ZImageTransformer2DModel.")

        super().__init__(
            all_patch_size=all_patch_size,
            all_f_patch_size=all_f_patch_size,
            in_channels=in_channels,
            dim=dim,
            n_layers=n_layers,
            n_refiner_layers=n_refiner_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            cap_feat_dim=cap_feat_dim,
            rope_theta=rope_theta,
            t_scale=t_scale,
            axes_dims=axes_dims,
            axes_lens=axes_lens,
        )

        # CachedTransformer compatibility
        self.do_true_cfg = False

        # NOTE: do NOT assign self.dtype — the diffusers-backed base class
        # (DiffusersZImageTransformer2DModel / ModelMixin) exposes `dtype` as a
        # read-only property derived from the parameters. Assigning it raises
        # "property 'dtype' has no setter". Weights load under
        # set_default_torch_dtype(od_config.dtype), so the inherited property
        # already returns the correct dtype.

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name.removeprefix("transformer.")
            if lookup_name not in params_dict:
                raise KeyError(f"Unknown Z-Image transformer weight: {original_name}")

            param = params_dict[lookup_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        return loaded_params


__all__ = [
    "FeedForward",
    "FinalLayer",
    "RopeEmbedder",
    "TimestepEmbedder",
    "ZImageTransformer2DModel",
    "ZImageTransformerBlock",
    "ZSingleStreamAttnProcessor",
    "validate_zimage_tp_constraints",
]
