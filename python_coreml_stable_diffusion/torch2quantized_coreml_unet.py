#!/usr/bin/env python
#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Quantize Stable Diffusion UNet and export to Core ML."""

import gc
import os
import torch

from python_coreml_stable_diffusion import torch2coreml
from python_coreml_stable_diffusion import torch2quantized_coreml_prepare as prepare


def main(args):
    os.makedirs(args.o, exist_ok=True)

    pipe = torch2coreml.get_pipeline(args)
    if torch.backends.mps.is_available():
        pipe.to(device="mps", dtype=torch.float32)
    elif torch.cuda.is_available():
        pipe.to(device="cuda", dtype=torch.float32)
    else:
        pipe.to(device="cpu", dtype=torch.float32)

    prepare.convert_quantized_unet(pipe, args)

    del pipe
    gc.collect()


if __name__ == "__main__":
    parser = prepare.parser_spec()
    args = parser.parse_args()
    args.convert_unet = True
    main(args)
