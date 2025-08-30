#!/usr/bin/env python
#
# For licensing see accompanying LICENSE.md file.
#
"""Convert a previously quantized SDXL UNet to Core ML."""

import argparse
import gc
import logging
import os
from collections import OrderedDict

import coremltools as ct
import numpy as np
import torch

from python_coreml_stable_diffusion import (
    torch2coreml,
    unet as unet_mod,
    chunk_mlprogram,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


torch.set_grad_enabled(False)


def convert_saved_quantized_unet(args):
    """Load a quantized UNet from disk and convert it to Core ML."""

    out_path = torch2coreml._get_out_path(args, "unet")
    unet_chunks_exist = all(
        os.path.exists(out_path.replace(".mlpackage", f"_chunk{idx+1}.mlpackage"))
        for idx in range(2)
    )

    if args.chunk_unet and unet_chunks_exist:
        logger.info("`unet` chunks already exist, skipping conversion.")
        return

    if os.path.exists(out_path):
        logger.info(f"`unet` already exists at {out_path}, skipping conversion.")
        return

    quant_dir = os.path.join(args.o, "sdxl_quantized_unet")
    quant_unet = unet_mod.UNet2DConditionModelXL.from_pretrained(quant_dir).eval()
    quant_unet.to("cpu")
    logger.info(f"Loaded quantized UNet from {quant_dir}")

    batch_size = 1 if args.unet_batch_one else 2
    sample_shape = (
        batch_size,
        quant_unet.config.in_channels,
        args.latent_h or quant_unet.config.sample_size,
        args.latent_w or quant_unet.config.sample_size,
    )

    encoder_hidden_states_shape = (
        batch_size,
        args.text_encoder_hidden_size or quant_unet.config.cross_attention_dim,
        1,
        args.text_token_sequence_length,
    )

    sample_inputs = OrderedDict(
        [
            ("sample", torch.rand(*sample_shape, dtype=torch.float32)),
            ("timestep", torch.tensor([0] * batch_size).to(torch.float32)),
            (
                "encoder_hidden_states",
                torch.rand(*encoder_hidden_states_shape, dtype=torch.float32),
            ),
        ]
    )

    height = (args.latent_h or quant_unet.config.sample_size) * 8
    width = (args.latent_w or quant_unet.config.sample_size) * 8
    original_size = (height, width)
    crops_coords_top_left = (0, 0)
    target_size = (height, width)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)
    time_ids = [add_neg_time_ids, add_time_ids]
    text_embeds_shape = (
        batch_size,
        args.text_encoder_2_hidden_size,
    )
    additional = OrderedDict(
        [
            ("time_ids", torch.tensor(time_ids).to(torch.float32)),
            (
                "text_embeds",
                torch.rand(*text_embeds_shape, dtype=torch.float32),
            ),
        ]
    )
    sample_inputs.update(additional)

    for k, v in sample_inputs.items():
        sample_inputs[k] = v.to(quant_unet.device)

    logger.info("JIT tracing quantized UNet")
    traced_unet = torch.jit.trace(quant_unet, list(sample_inputs.values()))

    coreml_inputs = {k: v.numpy().astype(np.float32) for k, v in sample_inputs.items()}
    coreml_unet, out_path = torch2coreml._convert_to_coreml(
        "unet",
        traced_unet,
        coreml_inputs,
        ["noise_pred"],
        args,
        precision=ct.precision.FLOAT32,
    )

    coreml_unet.author = f"Please refer to the Model Card available at huggingface.co/{args.model_version}"
    coreml_unet.license = "OpenRAIL++-M (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)"
    coreml_unet.version = args.model_version
    coreml_unet.short_description = (
        "Stable Diffusion generates images conditioned on text or other images as input through the diffusion process. "
        "Please refer to https://arxiv.org/abs/2112.10752 for details."
    )

    from python_coreml_stable_diffusion._version import __version__

    coreml_unet.user_defined_metadata[
        "com.github.apple.ml-stable-diffusion.version"
    ] = __version__

    coreml_unet.save(out_path)
    logger.info(f"Saved unet Core ML model to {out_path}")

    del quant_unet, traced_unet, coreml_unet
    gc.collect()

    if args.chunk_unet and not unet_chunks_exist:
        logger.info("Chunking unet in two approximately equal MLModels")
        args.mlpackage_path = out_path
        args.remove_original = False
        args.merge_chunks_in_pipeline_model = False
        chunk_mlprogram.main(args)


def main(args):
    os.makedirs(args.o, exist_ok=True)
    unet_mod.ATTENTION_IMPLEMENTATION_IN_EFFECT = unet_mod.AttentionImplementations[
        args.attention_implementation
    ]
    convert_saved_quantized_unet(args)


def parser_spec():
    parser = torch2coreml.parser_spec()
    parser.add_argument(
        "--text-encoder-2-hidden-size",
        type=int,
        default=1280,
        help="Hidden size for the second text encoder (SDXL).",
    )
    parser.set_defaults(
        text_token_sequence_length=77,
        text_encoder_hidden_size=2048,
    )
    return parser


if __name__ == "__main__":
    parser = parser_spec()
    args = parser.parse_args()
    args.convert_unet = True
    args.xl_version = True
    main(args)
