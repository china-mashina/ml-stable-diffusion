#!/usr/bin/env python
#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Quantize Stable Diffusion XL UNet and convert the pipeline to Core ML."""

import argparse
import gc
import logging
import os
import pickle
import json
import operator
from collections import OrderedDict
from copy import deepcopy

import coremltools as ct
import numpy as np
import torch
from tqdm import tqdm
from coremltools.optimize.torch.quantization import (
    LinearQuantizer,
    LinearQuantizerConfig,
    ModuleLinearQuantizerConfig,
)

from python_coreml_stable_diffusion import (
    torch2coreml,
    unet as unet_mod,
    chunk_mlprogram,
)
from python_coreml_stable_diffusion.layer_norm import LayerNormANE
from python_coreml_stable_diffusion.unet import Einsum

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _log_device(name, obj):
    """Utility to log the device of an object if available."""
    dev = None
    if hasattr(obj, "device"):
        dev = obj.device
    elif isinstance(obj, torch.nn.Module):
        try:
            dev = next(obj.parameters()).device
        except StopIteration:
            pass
    if dev is not None:
        logger.info(f"{name} device: {dev}")


torch.set_grad_enabled(False)

with open("prompts.txt", "r", encoding="utf-8") as f:
    CALIBRATION_DATA = [line.strip() for line in f.readlines()]

# Load negative prompts that mirror the calibration prompts.  Each line in
# ``negative_prompts.txt`` corresponds to the prompt on the same line in
# ``prompts.txt`` and is used during latent generation when running the
# pipeline.
with open("negative_prompts.txt", "r", encoding="utf-8") as f:
    NEGATIVE_CALIBRATION_DATA = [line.strip() for line in f.readlines()]

if len(NEGATIVE_CALIBRATION_DATA) != len(CALIBRATION_DATA):
    raise ValueError(
        "negative_prompts.txt must contain the same number of lines as prompts.txt"
    )


def register_input_log_hook(unet, inputs):
    """Register forward pre hook to save model inputs.

    Diffusers pipelines often call UNet with keyword arguments such as
    ``encoder_hidden_states`` (and ``time_ids``/``text_embeds`` for the XL
    variant).  The previous implementation only captured positional arguments
    and therefore missed these kwargs, resulting in incomplete calibration
    samples.  ``with_kwargs=True`` allows the hook to access both positional
    and keyword arguments so that all required inputs are logged.
    """

    def hook(_, args, kwargs):
        collected = list(args)
        for key in ["encoder_hidden_states", "time_ids", "text_embeds"]:
            if key in kwargs:
                collected.append(kwargs[key])
        if "added_cond_kwargs" in kwargs:
            added = kwargs["added_cond_kwargs"]
            for k in ["time_ids", "text_embeds"]:
                if isinstance(added, dict) and k in added:
                    collected.append(added[k])

        input_copy = tuple(
            i.detach().to("cpu") if torch.is_tensor(i) else i for i in collected
        )
        inputs.append(input_copy)

    return unet.register_forward_pre_hook(hook, with_kwargs=True)


def generate_calibration_data(pipe, args, calibration_dir):
    """Run prompts through pipeline and store UNet inputs for calibration."""

    unet_inputs = []
    handle = register_input_log_hook(pipe.unet, unet_inputs)

    pipe.scheduler.set_timesteps(4)
    pipe.scheduler.timesteps = torch.tensor(
        [999, 749, 499, 249], device=pipe.scheduler.timesteps.device
    )

    os.makedirs(calibration_dir, exist_ok=True)
    for f in os.listdir(calibration_dir):
        if f.endswith(".pkl"):
            os.remove(os.path.join(calibration_dir, f))

    prompts = (
        CALIBRATION_DATA if not getattr(args, "test", False) else CALIBRATION_DATA[:1]
    )
    negative_prompts = (
        NEGATIVE_CALIBRATION_DATA if not getattr(args, "test", False) else NEGATIVE_CALIBRATION_DATA[:1]
    )

    for prompt, negative_prompt in zip(prompts, negative_prompts):
        gen = torch.manual_seed(args.seed)
        pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=gen,
            num_inference_steps=4,
            guidance_scale=0,
        )
        filename = "_".join(prompt.split(" ")) + "_" + str(args.seed) + ".pkl"
        filepath = os.path.join(calibration_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(unet_inputs, f)
        unet_inputs.clear()

    handle.remove()


def unet_data_loader(data_dir, device="cpu", calibration_nsamples=None):
    """Load serialized UNet inputs from calibration directory."""

    dataloader = []
    skip_load = False
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".pkl"):
            filepath = os.path.join(data_dir, file)
            with open(filepath, "rb") as data:
                try:
                    while not skip_load:
                        unet_data = pickle.load(data)
                        for inp in unet_data:
                            dataloader.append(
                                [x.to(torch.float32).to(device) for x in inp]
                            )
                            for i, t in enumerate(dataloader[-1]):
                                _log_device(
                                    f"Loaded tensor device {len(dataloader)-1}_{i}", t
                                )
                            if (
                                calibration_nsamples
                                and len(dataloader) >= calibration_nsamples
                            ):
                                skip_load = True
                                break
                except EOFError:
                    pass
        if skip_load:
            break

    logger.info(f"Total calibration samples: {len(dataloader)}")
    return dataloader


def quantize_module_config(module_name):
    """Return config that quantizes only the specified module."""

    config = LinearQuantizerConfig(
        global_config=ModuleLinearQuantizerConfig(
            milestones=[0, 1000, 1000, 0],
            weight_dtype=torch.float32,
            activation_dtype=torch.float32,
        ),
        module_name_configs={
            module_name: ModuleLinearQuantizerConfig(
                quantization_scheme="symmetric",
                milestones=[0, 1000, 1000, 0],
            )
        },
        module_type_configs={
            torch.cat: None,
            torch.nn.GroupNorm: None,
            torch.nn.SiLU: None,
            torch.nn.functional.gelu: None,
            operator.add: None,
        },
    )
    return config


def get_quantizable_modules(unet):
    """Return list of leaf modules that can be quantized."""

    quantizable_modules = []
    for name, module in unet.named_modules():
        if len(list(module.children())) > 0:
            continue
        if isinstance(module, torch.nn.Conv2d):
            quantizable_modules.append(("conv", name))
        if isinstance(module, Einsum):
            quantizable_modules.append(("einsum", name))
    return quantizable_modules


def recipe_overrides_for_inference_speed(conv_layers, skipped_conv):
    """Force quantization for known compute hotspots."""

    for layer in conv_layers:
        # Quantize upsample stack convolutions
        if "upsamplers" in layer and layer in skipped_conv:
            logger.info(f"Removing {layer} from skip list")
            skipped_conv.remove(layer)
        # Quantize conv1 in up path resnets
        if (
            "up_blocks" in layer
            and "resnets" in layer
            and "conv1" in layer
            and layer in skipped_conv
        ):
            logger.info(f"Removing {layer} from skip list")
            skipped_conv.remove(layer)


def recipe_overrides_for_quality(conv_layers, einsum_layers, skipped_conv, skipped_einsum):
    """Skip layers that are sensitive to quantization artifacts."""

    for layer in conv_layers:
        # to_out projections from attention blocks
        if "to_out" in layer:
            skipped_conv.add(layer)
        # First and last UNet convolutions
        if layer in {"conv_in", "conv_out"}:
            skipped_conv.add(layer)
        # High resolution up path last resblocks
        if "up_blocks.3" in layer:
            skipped_conv.add(layer)
        # Mid-block attention convolutions
        if "mid_block.attentions" in layer:
            skipped_conv.add(layer)

    for layer in einsum_layers:
        if "mid_block.attentions" in layer:
            skipped_einsum.add(layer)


def quantize_cumulative_config(skip_conv_layers, skip_einsum_layers):
    """Return LinearQuantizerConfig for W8A8 quantization."""

    logger.info(
        f"Skipping {len(skip_conv_layers)} conv layers and {len(skip_einsum_layers)} einsum layers"
    )

    w8config = ModuleLinearQuantizerConfig(
        quantization_scheme="symmetric",
        milestones=[0, 1000, 1000, 0],
        activation_dtype=torch.float32,
    )

    conv_modules_config = {name: w8config for name in skip_conv_layers}
    einsum_modules_config = {name: w8config for name in skip_einsum_layers}
    module_name_config = {}
    module_name_config.update(conv_modules_config)
    module_name_config.update(einsum_modules_config)

    config = LinearQuantizerConfig(
        global_config=ModuleLinearQuantizerConfig(
            quantization_scheme="symmetric",
            milestones=[0, 1000, 1000, 0],
        ),
        module_name_configs=module_name_config,
        module_type_configs={
            torch.cat: None,
            torch.nn.GroupNorm: None,
            torch.nn.SiLU: None,
            torch.nn.functional.gelu: None,
            operator.add: None,
        },
    )
    return config


def _to_coreml_unet_inputs(
    sample,
    timestep,
    encoder_hidden_states,
    time_ids=None,
    text_embeds=None,
):
    """Normalize UNet inputs for Core ML conversion.

    ``time_ids`` and ``text_embeds`` are optional and used by SDXL models.
    They are forwarded as-is if provided.
    """

    if len(timestep.shape) == 0:
        timestep = timestep[None]
    timestep = timestep.expand(sample.shape[0])
    encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1).unsqueeze(2)

    inputs = [sample, timestep, encoder_hidden_states]
    if time_ids is not None or text_embeds is not None:
        inputs.extend([time_ids, text_embeds])
    return tuple(inputs)


def quantize(model, config, calibration_data):
    """Post training activation quantization using calibration data."""

    submodules = dict(model.named_modules(remove_duplicate=True))
    layer_norm_modules = [
        key for key, val in submodules.items() if isinstance(val, LayerNormANE)
    ]
    non_traceable = layer_norm_modules + ["time_proj", "time_embedding"]

    config.non_traceable_module_names = non_traceable
    config.preserved_attributes = ["config", "device"]

    sample_input = _to_coreml_unet_inputs(*calibration_data[0])
    for i, t in enumerate(sample_input):
        _log_device(f"Sample input tensor {i}", t)
    quantizer = LinearQuantizer(model, config)
    logger.info("Preparing model for quantization")
    prepared_model = quantizer.prepare(example_inputs=(sample_input,))
    device = sample_input[0].device
    prepared_model.to(device)
    _log_device("Prepared model", prepared_model)
    prepared_model.eval()

    quantizer.step()
    logger.info("Calibrate")
    for idx, data in enumerate(calibration_data):
        logger.info(f"Calibration data sample: {idx}")
        prepared_model(*_to_coreml_unet_inputs(*data))

    logger.info("Finalize model")
    quantized_model = quantizer.finalize()
    quantized_model.to(device)
    _log_device("Quantized model", quantized_model)
    return quantized_model


def _prepare_calibration(pipe, args, calib_dir):
    """Generate calibration data if needed and return dataloader."""
    if args.generate_calibration_data or not os.path.exists(calib_dir):
        logger.info("Generating calibration data for activation quantization")
        generate_calibration_data(pipe, args, calib_dir)
    # Quantization is only supported on CPU or single-device CUDA modules.
    # Loading calibration samples on CPU avoids device mismatch errors during
    # preparation.
    device = "cpu"
    dataloader = unet_data_loader(calib_dir, device, args.calibration_nsamples)
    if dataloader:
        for i, t in enumerate(dataloader[0]):
            _log_device(f"Calibration sample[0] tensor {i}", t)
    return dataloader


def register_input_preprocessing_hook(pipe):
    """Register hook to adapt pipeline inputs for the quantized UNet."""

    def hook(_, args, kwargs):
        sample = args[0]
        timestep = args[1]
        encoder_hidden_states = kwargs["encoder_hidden_states"]
        time_ids = kwargs.get("time_ids")
        text_embeds = kwargs.get("text_embeds")
        if "added_cond_kwargs" in kwargs:
            added = kwargs["added_cond_kwargs"]
            if isinstance(added, dict):
                time_ids = time_ids or added.get("time_ids")
                text_embeds = text_embeds or added.get("text_embeds")
        modified = _to_coreml_unet_inputs(
            sample, timestep, encoder_hidden_states, time_ids, text_embeds
        )
        return (modified, {})

    return pipe.unet.register_forward_pre_hook(hook, with_kwargs=True)


def _patch_add_embedding_linear_attrs(unet):
    """Provide Linear-like attributes for Conv-based timestep embeddings."""

    if hasattr(unet, "add_embedding"):
        lin1 = getattr(unet.add_embedding, "linear_1", None)
        if lin1 is not None and not hasattr(lin1, "in_features"):
            if hasattr(lin1, "in_channels"):
                lin1.in_features = lin1.in_channels
            elif hasattr(lin1, "conv") and hasattr(lin1.conv, "in_channels"):
                lin1.in_features = lin1.conv.in_channels


def prepare_pipe(pipe, unet):
    """Create a new pipeline with ``unet`` as the noise predictor."""

    new_pipe = deepcopy(pipe)
    unet.to(new_pipe.unet.device)
    new_pipe.unet = unet
    pre_hook_handle = register_input_preprocessing_hook(new_pipe)
    return new_pipe, pre_hook_handle


def run_pipe(pipe, prompts, negative_prompts, seed):
    gen = torch.manual_seed(seed)
    kwargs = dict(prompt=prompts, output_type="latent", generator=gen)
    if negative_prompts is not None:
        kwargs["negative_prompt"] = negative_prompts
        kwargs["guidance_scale"] = 0
    return np.array([latent.cpu().numpy() for latent in pipe(**kwargs).images])


def layerwise_sensitivity(pipe, args):
    """Compute layer-wise PSNR by quantizing one layer at a time."""

    calib_dir = os.path.join(args.o, "calibration_data")
    dataloader = _prepare_calibration(pipe, args, calib_dir)

    # Replace diffusers UNet with fx-traceable implementation before analysis
    if args.xl_version:
        unet_cls = unet_mod.UNet2DConditionModelXL
    else:
        unet_cls = unet_mod.UNet2DConditionModel
    reference_unet = unet_cls(
        support_controlnet=args.unet_support_controlnet, **pipe.unet.config
    ).eval()
    reference_unet.load_state_dict(pipe.unet.state_dict())
    _patch_add_embedding_linear_attrs(reference_unet)
    # ``prepare_pipe`` will move ``reference_unet`` to the pipeline's device, so we
    # keep it on CPU here to avoid an extra transfer when CUDA is available.
    pipe, _ = prepare_pipe(pipe, reference_unet)

    prompts = CALIBRATION_DATA if not getattr(args, "test", False) else CALIBRATION_DATA[:1]
    negative_prompts = (
        NEGATIVE_CALIBRATION_DATA
        if not getattr(args, "test", False)
        else NEGATIVE_CALIBRATION_DATA[:1]
    )

    logger.info("Generating reference outputs")
    ref_out = run_pipe(pipe, prompts, negative_prompts, args.seed)

    quantizable_modules = get_quantizable_modules(pipe.unet)
    results = {"conv": {}, "einsum": {}, "model_version": args.model_version}

    for module_type, module_name in tqdm(quantizable_modules):
        logger.info(f"Quantizing UNet layer: {module_name}")
        config = quantize_module_config(module_name)
        # Calibration samples are stored on CPU, so quantize a CPU copy of the
        # module and let ``prepare_pipe`` move it back to the pipeline's device
        # for evaluation.
        quantized_unet = quantize(deepcopy(pipe.unet).to("cpu"), config, dataloader)
        _patch_add_embedding_linear_attrs(quantized_unet)
        q_pipe, handle = prepare_pipe(pipe, quantized_unet)
        test_out = run_pipe(q_pipe, prompts, negative_prompts, args.seed)
        psnr = [float(f"{torch2coreml.compute_psnr(r, t):.1f}") for r, t in zip(ref_out, test_out)]
        avg_psnr = sum(psnr) / len(psnr)
        logger.info(f"AVG PSNR: {avg_psnr}")
        results[module_type][module_name] = avg_psnr
        handle.remove()
        del quantized_unet, q_pipe
        gc.collect()

    recipe_json_path = os.path.join(
        args.o, f"{args.model_version.replace('/', '_')}_quantization_recipe.json"
    )
    with open(recipe_json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Layer-wise sensitivity results saved to {recipe_json_path}")


def convert_quantized_unet(pipe, args):
    """Quantize ``pipe.unet`` and convert it to Core ML."""
    out_path = torch2coreml._get_out_path(args, "unet")

    unet_chunks_exist = all(
        os.path.exists(out_path.replace(".mlpackage", f"_chunk{idx+1}.mlpackage"))
        for idx in range(2)
    )

    if args.chunk_unet and unet_chunks_exist:
        logger.info("`unet` chunks already exist, skipping conversion.")
        del pipe.unet
        gc.collect()
        return

    if os.path.exists(out_path):
        logger.info(f"`unet` already exists at {out_path}, skipping conversion.")
        return

    # Calibration data
    calib_dir = os.path.join(
        args.o, f"calibration_data"
    )
    dataloader = _prepare_calibration(pipe, args, calib_dir)

    # Create a reference UNet and gather text encoder metadata before
    # releasing the pipeline to free memory.
    if args.xl_version:
        unet_cls = unet_mod.UNet2DConditionModelXL
    else:
        unet_cls = unet_mod.UNet2DConditionModel

    # Move pipeline UNet to CPU before collecting weights to avoid mixed
    # device tensors in the reference model.
    pipe.unet.to("cpu")

    reference_unet = unet_cls(
        support_controlnet=args.unet_support_controlnet, **pipe.unet.config
    ).eval()
    reference_unet.load_state_dict(pipe.unet.state_dict())
    reference_unet.to("cpu")

    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        text_token_sequence_length = pipe.text_encoder.config.max_position_embeddings
        hidden_size = pipe.text_encoder.config.hidden_size
        if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
            te2_hidden_size = pipe.text_encoder_2.config.hidden_size
            del pipe.text_encoder_2
        else:
            te2_hidden_size = None
        del pipe.text_encoder
    else:
        text_token_sequence_length = pipe.text_encoder_2.config.max_position_embeddings
        hidden_size = pipe.text_encoder_2.config.hidden_size
        te2_hidden_size = pipe.text_encoder_2.config.hidden_size
        del pipe.text_encoder_2

    # Delete pipeline UNet and drop the pipeline reference to free memory before
    # quantization. The caller should ensure there are no remaining references
    # to the pipeline after this call.
    del pipe.unet
    del pipe
    gc.collect()

    # Quantization recipe from layerwise sensitivity
    recipe_json_path = os.path.join(
        args.o, f"{args.model_version.replace('/', '_')}_quantization_recipe.json"
    )
    if os.path.exists(recipe_json_path):
        with open(recipe_json_path, "r") as f:
            results = json.load(f)
        skipped_conv = {
            layer for layer, psnr in results["conv"].items() if psnr < args.conv_psnr
        }
        skipped_einsum = set()
        for layer, psnr in results["einsum"].items():
            if "attn" in layer or "attention" in layer:
                thresh = args.attn_psnr
            else:
                thresh = args.mlp_psnr
            if psnr < thresh:
                skipped_einsum.add(layer)
        conv_layers = list(results["conv"].keys())
        einsum_layers = list(results["einsum"].keys())
        recipe_overrides_for_inference_speed(conv_layers, skipped_conv)
        recipe_overrides_for_quality(
            conv_layers, einsum_layers, skipped_conv, skipped_einsum
        )
    else:
        logger.warning("Quantization recipe not found. Quantizing all layers.")
        skipped_conv = set()
        skipped_einsum = set()

    # Quantize UNet weights and activations (W8A8 by default)
    config = quantize_cumulative_config(skipped_conv, skipped_einsum)
    logger.info("Quantizing UNet model")

    # Quantization must run on CPU. Keeping the resulting model on the same
    # device avoids mixed-device errors when tracing.
    quant_device = "cpu"

    reference_unet.to(quant_device)
    _log_device("Reference UNet", reference_unet)

    quant_unet = quantize(reference_unet, config, dataloader)
    # reference_unet and calibration data are no longer needed
    del reference_unet, dataloader
    gc.collect()
    _log_device("Quantized UNet", quant_unet)

    # Prepare sample input shapes
    batch_size = 1 if args.unet_batch_one else 2
    sample_shape = (
        batch_size,
        quant_unet.config.in_channels,
        args.latent_h or quant_unet.config.sample_size,
        args.latent_w or quant_unet.config.sample_size,
    )


    encoder_hidden_states_shape = (
        batch_size,
        args.text_encoder_hidden_size
        or quant_unet.config.cross_attention_dim
        or hidden_size,
        1,
        args.text_token_sequence_length or text_token_sequence_length,
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

    if args.xl_version:
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
            te2_hidden_size,
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

    # Ensure sample inputs are on the same device as the model
    for k, v in sample_inputs.items():
        sample_inputs[k] = v.to(quant_unet.device)
        _log_device(f"Sample input {k}", sample_inputs[k])

    sample_inputs_spec = {k: (v.shape, v.dtype) for k, v in sample_inputs.items()}
    logger.info(f"Sample UNet inputs spec: {sample_inputs_spec}")

    logger.info("JIT tracing quantized UNet")
    traced_unet = torch.jit.trace(quant_unet, list(sample_inputs.values()))
    _log_device("Traced UNet", traced_unet)

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
    if args.xl_version:
        coreml_unet.license = "OpenRAIL++-M (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)"
    else:
        coreml_unet.license = (
            "OpenRAIL (https://huggingface.co/spaces/CompVis/stable-diffusion-license)"
        )
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
    logger.info(f"Saved quantized unet into {out_path}")

    del quant_unet, traced_unet, coreml_unet
    gc.collect()

    if args.chunk_unet and not unet_chunks_exist:
        logger.info("Chunking unet in two approximately equal MLModels")
        args.mlpackage_path = out_path
        args.remove_original = False
        args.merge_chunks_in_pipeline_model = False
        chunk_mlprogram.main(args)

def chunk_unet(args):
    out_path = torch2coreml._get_out_path(args, "unet")
    unet_chunks_exist = all(
        os.path.exists(out_path.replace(".mlpackage", f"_chunk{idx+1}.mlpackage"))
        for idx in range(2)
    )
    if not unet_chunks_exist:
        logger.info("Chunking unet in two approximately equal MLModels")
        args.mlpackage_path = out_path
        args.remove_original = False
        args.merge_chunks_in_pipeline_model = False
        chunk_mlprogram.main(args)
        
def main(args):
    os.makedirs(args.o, exist_ok=True)

    # Load diffusers pipeline as in ``torch2coreml``.  Run the pipeline on CUDA
    # when available to speed up layer-wise sensitivity sweeps.
    pipe = torch2coreml.get_pipeline(args)
    if torch.cuda.is_available():
        pipe.to(device="cuda", dtype=torch.float32)
    elif torch.backends.mps.is_available():
        pipe.to(device="mps", dtype=torch.float32)
    else:
        pipe.to(device="cpu", dtype=torch.float32)
    _log_device("Pipeline", pipe)
    _log_device("UNet", pipe.unet)
    if getattr(pipe, "text_encoder", None) is not None:
        _log_device("TextEncoder", pipe.text_encoder)
    if getattr(pipe, "text_encoder_2", None) is not None:
        _log_device("TextEncoder2", pipe.text_encoder_2)
    if getattr(pipe, "vae", None) is not None:
        _log_device("VAE", pipe.vae)

    unet_mod.ATTENTION_IMPLEMENTATION_IN_EFFECT = unet_mod.AttentionImplementations[
        args.attention_implementation
    ]

    if args.layerwise_sensitivity:
        layerwise_sensitivity(pipe, args)

    if args.convert_unet:
        convert_quantized_unet(pipe, args)
        del pipe
        gc.collect()
    
    if args.chunk_unet:
        chunk_unet(args)


def parser_spec():
    parser = torch2coreml.parser_spec()
    parser.add_argument(
        "--generate-calibration-data",
        action="store_true",
        help="Generate calibration data for activation quantization",
    )
    parser.add_argument(
        "--calibration-nsamples",
        type=int,
        default=None,
        help="Number of samples for calibration",
    )
    parser.add_argument(
        "--seed", "-s", default=50, type=int, help="Random seed for calibration prompts"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run calibration data generation on a single prompt",
    )
    parser.add_argument(
        "--layerwise-sensitivity",
        action="store_true",
        help="Compute compression sensitivity per-layer, by quantizing one layer at a time",
    )
    parser.add_argument(
        "--conv-psnr",
        type=float,
        default=42.0,
        help="PSNR threshold for convolution layers",
    )
    parser.add_argument(
        "--attn-psnr",
        type=float,
        default=35.0,
        help="PSNR threshold for attention linear layers",
    )
    parser.add_argument(
        "--mlp-psnr",
        type=float,
        default=36.0,
        help="PSNR threshold for MLP linear layers",
    )
    return parser


