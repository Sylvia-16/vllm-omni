# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def _make_tiny_transformer():
    from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel

    return FluxTransformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        guidance_embeds=True,
        axes_dims_rope=(2, 2, 4),
    )


def test_flux_transformer_uses_diffusers_implementation():
    from diffusers.models.transformers.transformer_flux import (
        FluxTransformer2DModel as DiffusersFluxTransformer2DModel,
    )

    transformer = _make_tiny_transformer()

    assert isinstance(transformer, DiffusersFluxTransformer2DModel)
    assert transformer.in_channels == transformer.config.in_channels == 16
    assert transformer.guidance_embeds is transformer.config.guidance_embeds is True
    assert transformer._layerwise_offload_blocks_attrs == [
        "transformer_blocks",
        "single_transformer_blocks",
    ]


def test_flux_transformer_loads_prefixed_diffusers_weights_and_forwards():
    from vllm.model_executor.models.utils import AutoWeightsLoader

    transformer = _make_tiny_transformer()
    pipeline = torch.nn.Module()
    pipeline.add_module("transformer", transformer)
    prefixed_weights = [
        (f"transformer.{name}", parameter.detach().clone())
        for name, parameter in transformer.named_parameters()
    ]

    loaded_names = AutoWeightsLoader(pipeline).load_weights(prefixed_weights)
    assert {f"transformer.{name}" for name, _ in transformer.named_parameters()} <= loaded_names

    output = transformer(
        hidden_states=torch.randn(1, 4, 16),
        encoder_hidden_states=torch.randn(1, 3, 16),
        pooled_projections=torch.randn(1, 8),
        timestep=torch.tensor([0.5]),
        img_ids=torch.zeros(4, 3),
        txt_ids=torch.zeros(3, 3),
        guidance=torch.tensor([3.5]),
        return_dict=False,
    )

    assert output[0].shape == (1, 4, 16)


def test_hunyuan_video_keeps_flux_feed_forward_import():
    from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
        HunyuanVideo15Transformer3DModel,
    )

    assert HunyuanVideo15Transformer3DModel is not None


def test_flux_pipeline_creates_guidance_on_latent_device():
    from vllm_omni.diffusion.models.flux.pipeline_flux import prepare_guidance

    latents = torch.zeros(3, 4, 16)
    guidance = prepare_guidance(3.5, latents)

    assert guidance.device == latents.device
    assert guidance.dtype == torch.float32
    assert guidance.tolist() == [3.5, 3.5, 3.5]
