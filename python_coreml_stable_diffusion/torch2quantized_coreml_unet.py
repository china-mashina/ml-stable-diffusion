#!/usr/bin/env python
#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Quantize Stable Diffusion UNet and export to Core ML."""

import os

from python_coreml_stable_diffusion import torch2quantized_coreml_prepare as prepare


def main(args):
    os.makedirs(args.o, exist_ok=True)

    prepare.main(args)


if __name__ == "__main__":
    parser = prepare.parser_spec()
    args = parser.parse_args()
    args.convert_unet = True
    main(args)
