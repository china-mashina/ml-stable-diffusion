#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
from collections import OrderedDict

import coremltools as ct
from coremltools.converters.mil import Block, Program, Var
from coremltools.converters.mil.frontend.milproto.load import load as _milproto_to_pymil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Placeholder
from coremltools.converters.mil.mil import types as types
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.testing_utils import random_gen_input_feature_type

import gc

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import os
from python_coreml_stable_diffusion import torch2coreml
import shutil
import time


def _verify_output_correctness_of_chunks(full_model,
                                         first_chunk_model=None,
                                         second_chunk_model=None,
                                         pipeline_model=None,):
    """ Verifies the end-to-end output correctness of full (original) model versus chunked models
    """
    # Generate inputs for first chunk and full model
    input_dict = {}
    for input_desc in full_model._spec.description.input:
        input_dict[input_desc.name] = random_gen_input_feature_type(input_desc)

    # Generate outputs for full model
    outputs_from_full_model = full_model.predict(input_dict)

    if pipeline_model is not None:
        outputs_from_pipeline_model = pipeline_model.predict(input_dict)
        final_outputs = outputs_from_pipeline_model

    elif first_chunk_model is not None and second_chunk_model is not None:
        # Generate outputs for first chunk
        outputs_from_first_chunk_model = first_chunk_model.predict(input_dict)

        # Prepare inputs for second chunk model from first chunk's outputs and regular inputs
        second_chunk_input_dict = {}
        for input_desc in second_chunk_model._spec.description.input:
            if input_desc.name in outputs_from_first_chunk_model:
                second_chunk_input_dict[
                    input_desc.name] = outputs_from_first_chunk_model[
                        input_desc.name]
            else:
                second_chunk_input_dict[input_desc.name] = input_dict[
                    input_desc.name]

        # Generate output for second chunk model
        outputs_from_second_chunk_model = second_chunk_model.predict(
            second_chunk_input_dict)
        final_outputs = outputs_from_second_chunk_model
    else:
        raise ValueError

    # Verify correctness across all outputs from second chunk and full model
    for out_name in outputs_from_full_model.keys():
        torch2coreml.report_correctness(
            original_outputs=outputs_from_full_model[out_name],
            final_outputs=final_outputs[out_name],
            log_prefix=f"{out_name}")


def _load_prog_from_mlmodel(model):
    """ Load MIL Program from an MLModel
    """
    model_spec = model.get_spec()
    start_ = time.time()
    logger.info(
        "Loading MLModel object into a MIL Program object (including the weights).."
    )
    prog = _milproto_to_pymil(
        model_spec=model_spec,
        specification_version=model_spec.specificationVersion,
        file_weights_dir=model.weights_dir,
    )
    logger.info(f"Program loaded in {time.time() - start_:.1f} seconds")

    return prog


def _get_op_idx_split_location(prog: Program):
    """Find the op that approximately bisects the graph by weight size.

    Quantized models often encode parameters via ``constexpr_*`` operations
    rather than plain ``const`` ops.  The original implementation ignored these
    ops which caused the computed split location to be heavily skewed towards
    the beginning of the graph.  This function accounts for both storage
    mechanisms by attributing the weight size to ``constexpr_*`` outputs and
    skipping ``const`` ops that only feed such operators.
    """

    main_block = prog.functions["main"]
    main_block.operations = list(main_block.operations)

    def _tensor_size_in_mb(var: Var) -> float:
        """Return the memory footprint of ``var`` in megabytes."""

        # Depending on how the tensor was created, ``var.val`` could either be a
        # ``numpy.ndarray`` directly or a wrapper object with a ``val`` attribute
        # pointing to the array.  Older implementations assumed the latter which
        # causes an ``AttributeError`` when ``var.val`` is already an ndarray.
        val = var.val
        if isinstance(val, np.ndarray):
            arr = val
        elif val is not None and hasattr(val, "val") and isinstance(val.val, np.ndarray):
            arr = val.val
        else:
            arr = None

        if arr is not None:
            return arr.size * arr.itemsize / (1024 * 1024)

        if (
            var.shape is not None
            and var.dtype is not None
            and all(dim is not None for dim in var.shape)
        ):
            np_dtype = types.nptype_from_builtin(var.dtype)
            return (np.prod(var.shape) * np.dtype(np_dtype).itemsize) / (1024 * 1024)

        return 0.0

    accounted_ops = set()

    def _collect_const_size(var: Var) -> float:
        """Return the size of a constant ancestor and mark it as accounted."""
        if not isinstance(var, Var) or var.op is None:
            return 0.0

        producer = var.op
        if producer in accounted_ops:
            return 0.0

        if producer.op_type == "const" or producer.op_type.startswith("constexpr_"):
            accounted_ops.add(producer)
            return _tensor_size_in_mb(producer.outputs[0])

        if producer.op_type in {"quantize", "dequantize"}:
            inp = list(producer.inputs.values())[0]
            if isinstance(inp, (list, tuple)):
                inp = inp[0]
            return _collect_const_size(inp)

        return 0.0

    def _op_weight_size_in_mb(op) -> float:
        size = 0.0
        for inp in op.inputs.values():
            if isinstance(inp, (list, tuple)):
                for v in inp:
                    size += _collect_const_size(v)
            else:
                size += _collect_const_size(inp)
        return size

    op_sizes = [
        _op_weight_size_in_mb(op) for op in main_block.operations
    ]
    total_size_in_mb = sum(op_sizes)
    half_size = total_size_in_mb * 0.625

    cumulative_size_in_mb = 0.0
    for op, size in zip(main_block.operations, op_sizes):
        cumulative_size_in_mb += size

        # Select a non-const op (single child) once the cumulative size exceeds half
        if (cumulative_size_in_mb > half_size and not op.op_type.startswith("const")
                and len(op.outputs) == 1
                and len(op.outputs[0].child_ops) == 1):
            op_idx = main_block.operations.index(op)
            return op_idx, cumulative_size_in_mb, total_size_in_mb


def _get_first_chunk_outputs(block, op_idx):
    # Get the list of all vars that go across from first program (all ops from 0 to op_idx (inclusive))
    # to the second program (all ops from op_idx+1 till the end). These all vars need to be made the output
    # of the first program and the input of the second program
    boundary_vars = set()
    block.operations = list(block.operations)
    for i in range(op_idx + 1):
        op = block.operations[i]
        if not op.op_type.startswith("const"):
            for var in op.outputs:
                if var.val is None:  # only consider non const vars
                    for child_op in var.child_ops:
                        child_op_idx = block.operations.index(child_op)
                        if child_op_idx > op_idx:
                            boundary_vars.add(var)
    return list(boundary_vars)


@block_context_manager
def _add_fp32_casts(block, boundary_vars):
    new_boundary_vars = []
    for var in boundary_vars:
        if var.dtype != types.fp16:
            new_boundary_vars.append(var)
        else:
            fp32_var = mb.cast(x=var, dtype="fp32", name=var.name)
            new_boundary_vars.append(fp32_var)
    return new_boundary_vars


def _make_first_chunk_prog(prog, op_idx):
    """ Build first chunk by declaring early outputs and removing unused subgraph
    """
    block = prog.functions["main"]
    boundary_vars = _get_first_chunk_outputs(block, op_idx)

    # Due to possible numerical issues, cast any fp16 var to fp32
    new_boundary_vars = _add_fp32_casts(block, boundary_vars)

    block.outputs.clear()
    block.set_outputs(new_boundary_vars)
    PASS_REGISTRY["common::dead_code_elimination"](prog)
    return prog


def _make_second_chunk_prog(prog, op_idx):
    """ Build second chunk by rebuilding a pristine MIL Program from MLModel
    """
    block = prog.functions["main"]
    block.opset_version = ct.target.iOS17

    # First chunk outputs are second chunk inputs (e.g. skip connections)
    boundary_vars = _get_first_chunk_outputs(block, op_idx)

    # This op will not be included in this program. Its output var will be made into an input
    block.operations = list(block.operations)
    boundary_op = block.operations[op_idx]

    # Add all boundary ops as inputs
    with block:
        for var in boundary_vars:
            new_placeholder = Placeholder(
                sym_shape=var.shape,
                dtype=var.dtype if var.dtype != types.fp16 else types.fp32,
                name=var.name,
            )

            block._input_dict[
                new_placeholder.outputs[0].name] = new_placeholder.outputs[0]

            block.function_inputs = tuple(block._input_dict.values())
            new_var = None
            if var.dtype == types.fp16:
                new_var = mb.cast(x=new_placeholder.outputs[0],
                                  dtype="fp16",
                                  before_op=var.op)
            else:
                new_var = new_placeholder.outputs[0]

            block.replace_uses_of_var_after_op(
                anchor_op=boundary_op,
                old_var=var,
                new_var=new_var,
                # This is needed if the program contains "constexpr_*" ops. In normal cases, there are stricter
                # rules for removing them, and their presence may prevent replacing this var.
                # However in this case, since we want to remove all the ops in chunk 1, we can safely
                # set this to True.
                force_replace=True,
            )

    PASS_REGISTRY["common::dead_code_elimination"](prog)

    # Remove any unused inputs
    new_input_dict = OrderedDict()
    for k, v in block._input_dict.items():
        if len(v.child_ops) > 0:
            new_input_dict[k] = v
    block._input_dict = new_input_dict
    block.function_inputs = tuple(block._input_dict.values())

    return prog


def _legacy_model_chunking(args):
    # TODO: Remove this method after setting the coremltools dependency >= 8.0
    os.makedirs(args.o, exist_ok=True)

    # Check filename extension
    mlpackage_name = os.path.basename(args.mlpackage_path)
    name, ext = os.path.splitext(mlpackage_name)
    assert ext == ".mlpackage", f"`--mlpackage-path` (args.mlpackage_path) is not an .mlpackage file"

    # Load CoreML model
    logger.info("Loading model from {}".format(args.mlpackage_path))
    start_ = time.time()
    model = ct.models.MLModel(
        args.mlpackage_path,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    logger.info(
        f"Loading {args.mlpackage_path} took {time.time() - start_:.1f} seconds"
    )

    # Load the MIL Program from MLModel
    prog = _load_prog_from_mlmodel(model)

    # Compute the incision point by bisecting the program based on weights size
    op_idx, first_chunk_weights_size, total_weights_size = _get_op_idx_split_location(
        prog)
    main_block = prog.functions["main"]
    incision_op = main_block.operations[op_idx]
    logger.info(f"{args.mlpackage_path} will chunked into two pieces.")
    logger.info(
        f"The incision op: name={incision_op.name}, type={incision_op.op_type}, index={op_idx}/{len(main_block.operations)}"
    )
    logger.info(f"First  chunk size = {first_chunk_weights_size:.2f} MB")
    logger.info(
        f"Second chunk size = {total_weights_size - first_chunk_weights_size:.2f} MB"
    )

    # Build first chunk (in-place modifies prog by declaring early exits and removing unused subgraph)
    prog_chunk1 = _make_first_chunk_prog(prog, op_idx)

    # Build the second chunk
    prog_chunk2 = _make_second_chunk_prog(_load_prog_from_mlmodel(model),
                                          op_idx)

    if not args.check_output_correctness:
        # Original model no longer needed in memory
        del model
        gc.collect()

    # Convert the MIL Program objects into MLModels
    logger.info("Converting the two programs")
    model_chunk1 = ct.convert(
        prog_chunk1,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS17,
    )
    del prog_chunk1
    gc.collect()
    logger.info("Conversion of first chunk done.")

    model_chunk2 = ct.convert(
        prog_chunk2,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS17,
    )
    del prog_chunk2
    gc.collect()
    logger.info("Conversion of second chunk done.")

    # Verify output correctness
    if args.check_output_correctness:
        logger.info("Verifying output correctness of chunks")
        _verify_output_correctness_of_chunks(
            full_model=model,
            first_chunk_model=model_chunk1,
            second_chunk_model=model_chunk2,
        )

    if args.merge_chunks_in_pipeline_model:
        # Make a single pipeline model to manage the model chunks
        pipeline_model = ct.utils.make_pipeline(model_chunk1, model_chunk2)
        out_path_pipeline = os.path.join(args.o, name + "_chunked_pipeline.mlpackage")

        # Save and reload to ensure CPU placement
        pipeline_model.save(out_path_pipeline)
        pipeline_model = ct.models.MLModel(out_path_pipeline, compute_units=ct.ComputeUnit.CPU_ONLY)

        if args.check_output_correctness:
            logger.info("Verifying output correctness of pipeline model")
            _verify_output_correctness_of_chunks(
                full_model=model,
                pipeline_model=pipeline_model,
            )
    else:
        # Save the chunked models to disk
        out_path_chunk1 = os.path.join(args.o, name + "_chunk1.mlpackage")
        out_path_chunk2 = os.path.join(args.o, name + "_chunk2.mlpackage")

        logger.info(
            f"Saved chunks in {args.o} with the suffix _chunk1.mlpackage and _chunk2.mlpackage"
        )
        model_chunk1.save(out_path_chunk1)
        model_chunk2.save(out_path_chunk2)
        logger.info("Done.")


def main(args):
    ct_version = ct.__version__

    # With coremltools version <= 8.0b1,
    # we use the legacy implementation.
    # TODO: Remove the logic after setting the coremltools dependency >= 8.0.
    logger.info(
        f"coremltools version {ct_version} detected. Recommended upgrading the package version to "
        f"'8.0b2' when you running chunk_mlprogram.py script for the latest supports and bug fixes."
    )
    _legacy_model_chunking(args)

    # Remove original (non-chunked) model if requested
    if args.remove_original:
        logger.info(
            "Removing original (non-chunked) model at {args.mlpackage_path}")
        shutil.rmtree(args.mlpackage_path)
        logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlpackage-path",
        required=True,
        help=
        "Path to the mlpackage file to be split into two mlpackages of approximately same file size.",
    )
    parser.add_argument(
        "-o",
        required=True,
        help=
        "Path to output directory where the two model chunks should be saved.",
    )
    parser.add_argument(
        "--remove-original",
        action="store_true",
        help=
        "If specified, removes the original (non-chunked) model to avoid duplicating storage."
    )
    parser.add_argument(
        "--check-output-correctness",
        action="store_true",
        help=
        ("If specified, compares the outputs of original Core ML model with that of pipelined CoreML model chunks and reports PSNR in dB. ",
         "Enabling this feature uses more memory. Disable it if your machine runs out of memory."
         ))
    parser.add_argument(
        "--merge-chunks-in-pipeline-model",
        action="store_true",
        help=
        ("If specified, model chunks are managed inside a single pipeline model for easier asset maintenance"
         ))

    args = parser.parse_args()
    main(args)
