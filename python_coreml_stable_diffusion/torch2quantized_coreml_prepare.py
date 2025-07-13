#!/usr/bin/env python
"""Convert quantized Stable Diffusion XL models to Core ML.

This helper mirrors ``torch2coreml`` but uses quantized UNet weights
produced by :mod:`activation_quantization` and automatically splits the
UNet into two chunks when ``--chunk-unet`` is specified.
"""

import argparse
import os
import gc
import logging
from types import SimpleNamespace

import coremltools as ct

from python_coreml_stable_diffusion import (
    torch2coreml,
    torch2quantized_coreml as qcoreml,
    chunk_mlprogram,
    unet as unet_mod,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_quantized_unet_with_chunking(pipe, args):
    """Quantize the pipeline's UNet and export to Core ML.

    After conversion the resulting model is optionally split into two
    mlpackages for mobile deployment.
    """
    qcoreml.convert_quantized_unet(pipe, args)

    if not args.chunk_unet:
        return

    out_path = torch2coreml._get_out_path(args, "unet")
    chunk_args = SimpleNamespace(
        mlpackage_path=out_path,
        o=args.o,
        remove_original=False,
        check_output_correctness=args.check_output_correctness,
        merge_chunks_in_pipeline_model=False,
    )
    if os.path.exists(out_path):
        logger.info("Chunking unet model into two mlpackages")
        chunk_mlprogram.main(chunk_args)


def main(args):
    os.makedirs(args.o, exist_ok=True)

    pipe = torch2coreml.get_pipeline(args)

    unet_mod.ATTENTION_IMPLEMENTATION_IN_EFFECT = unet_mod.AttentionImplementations[
        args.attention_implementation
    ]

    if args.convert_vae_decoder:
        torch2coreml.convert_vae_decoder(pipe, args)
    if args.convert_vae_encoder:
        torch2coreml.convert_vae_encoder(pipe, args)
    if args.convert_text_encoder and hasattr(pipe, "text_encoder"):
        torch2coreml.convert_text_encoder(
            pipe.text_encoder, pipe.tokenizer, "text_encoder", args
        )
    if args.convert_text_encoder and hasattr(pipe, "text_encoder_2"):
        torch2coreml.convert_text_encoder(
            pipe.text_encoder_2, pipe.tokenizer_2, "text_encoder_2", args
        )
    if args.convert_safety_checker:
        torch2coreml.convert_safety_checker(pipe, args)

    if args.convert_unet:
        convert_quantized_unet_with_chunking(pipe, args)

    if args.quantize_nbits is not None:
        torch2coreml.quantize_weights(args)

    if args.bundle_resources_for_swift_cli:
        torch2coreml.bundle_resources_for_swift_cli(args)



def parser_spec():
    parser = qcoreml.parser_spec()
    parser.add_argument(
        "--chunk-unet",
        action="store_true",
        help="Split the quantized UNet into two mlpackages after conversion",
    )
    return parser


if __name__ == "__main__":
    parser = parser_spec()
    args = parser.parse_args()
    main(args)
