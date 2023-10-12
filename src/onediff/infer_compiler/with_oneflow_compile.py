from . import obj_1f_from_torch
from .obj_1f_from_torch import _get_module
import os
import oneflow as flow
import torch


def get_unet_graph(size=9):
    class UNetGraph(flow.nn.Graph):
        @flow.nn.Graph.with_dynamic_input_shape(size=size)
        def __init__(self, unet):
            super().__init__(enable_get_runtime_state_dict=True)
            self.unet = unet
            self.config.enable_cudnn_conv_heuristic_search_algo(False)
            self.config.allow_fuse_add_to_output(True)

            os.environ["ONEFLOW_GRAPH_DELAY_VARIABLE_OP_EXECUTION"] = "1"
            os.environ["ONEFLOW_MLIR_CSE"] = "1"
            os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
            os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
            os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
            os.environ["ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL"] = "1"
            os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
            os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
            os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
            os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
            os.environ["ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
            os.environ["ONEFLOW_KERNEL_GEMM_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
            os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
            os.environ["ONEFLOW_KERNEL_GEMM_ENABLE_CUTLASS_IMPL"] = "1"
            os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
            os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
            os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"
            os.environ["ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL"] = "0"
            os.environ["ONEFLOW_MLIR_GROUP_MATMUL_QUANT"] = "1"
            # TODO: enable this will cause the failure of multi resolution warmup
            # os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
            # os.environ["ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH"] = "1"

        def build(self, *args, **kwargs):
            return self.unet(*args, **kwargs)

        def warmup_with_load(self, file_path, device=None):
            state_dict = flow.load(file_path)
            if device is not None:
                state_dict = flow.nn.Graph.runtime_state_dict_to(state_dict, device)
            self.load_runtime_state_dict(state_dict)

        def save_graph(self, file_path):
            state_dict = self.runtime_state_dict()
            flow.save(state_dict, file_path)

    return UNetGraph


def oneflow_compile(torch_unet, *, use_graph=True, options={}):
    torch2flow = {}
    of_md = _get_module(torch_unet, torch2flow)
    from oneflow.framework.args_tree import ArgsTree

    def input_fn(value):
        if isinstance(value, torch.Tensor):
            return flow.utils.tensor.from_torch(value)
        else:
            return value

    def output_fn(value):
        if isinstance(value, flow.Tensor):
            return flow.utils.tensor.to_torch(value)
        else:
            return value

    if use_graph:
        if "size" in options:
            size = options["size"]
        else:
            size = 9
        dpl_graph = get_unet_graph(size)(of_md)

    class DeplayableModule(of_md.__class__):
        def __call__(self, *args, **kwargs):
            args_tree = ArgsTree((args, kwargs), False, tensor_type=torch.Tensor)
            out = args_tree.map_leaf(input_fn)
            mapped_args = out[0]
            mapped_kwargs = out[1]

            if use_graph:
                output = self._dpl_graph(*mapped_args, **mapped_kwargs)
            else:
                output = super().__call__(*mapped_args, **mapped_kwargs)

            out_tree = ArgsTree((output, None), False)
            out = out_tree.map_leaf(output_fn)
            return out[0]

        def warmup_with_load(self, file_path, device=None):
            self._dpl_graph.warmup_with_load(file_path, device)

        def save_graph(self, file_path):
            self._dpl_graph.save_graph(file_path)

    of_md.__class__ = DeplayableModule
    if use_graph:
        of_md._dpl_graph = dpl_graph
    return of_md
