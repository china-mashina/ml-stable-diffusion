#!/usr/bin/env python
#
# For licensing see accompanying LICENSE.md file.
#
"""Convert Stable Diffusion XL pipeline to Core ML with quantized UNet.

This script wraps ``python_coreml_stable_diffusion.torch2quantized_coreml_prepare``
and runs it with preset arguments so it can be executed with no command line
options."""

from python_coreml_stable_diffusion import torch2quantized_coreml_prepare as prepare


DEFAULT_ARGS = [
    "--convert-unet",
    "--xl-version",
    "--model-version",
    "model",
    "--attention-implementation",
    "SPLIT_EINSUM",
    "-o",
    "output",
    "--generate-calibration-data",
    "--min-deployment-target",
    "iOS17",
    "--latent-h",
    "96",
    "--latent-w",
    "96",
    "--chunk-unet"
]


def main():
    parser = prepare.parser_spec()
    args = parser.parse_args(DEFAULT_ARGS)
    prepare.main(args)


if __name__ == "__main__":
    main()
