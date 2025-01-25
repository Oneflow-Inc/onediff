import os
import warnings
from dataclasses import dataclass
from functools import partial, singledispatchmethod
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from comfy.controlnet import ControlNet
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.backends.oneflow import (
    OneflowDeployableModule as DeployableModule,
)
from onediff.optimization import quant_optimizer
from onediff_quant.quantization import QuantizationConfig
from onediff_quant.quantization.module_operations import get_sub_module
from onediff_quant.quantization.quantize_calibrators import (
    QuantizationMetricsCalculator,
)
from onediff_quant.quantization.quantize_config import Metric

from .booster_basic import BoosterExecutor
from .patch_management import create_patch_executor, PatchType
from .utils.booster_utils import (
    is_fp16_model,
    set_compiled_options,
    set_environment_for_svd_img2vid,
)
from .utils.graph_path import generate_graph_path


class SubQuantizationPercentileCalculator(QuantizationMetricsCalculator):
    def __init__(
        self,
        model: torch.nn.Module,
        config: QuantizationConfig,
        cache_key: str = "",
        module_selector: callable = lambda x: x,
        *,
        seed=1,
        select=Metric.MAE.value,
        conv_percentage=0.9,
        linear_percentage=0.9,
    ):
        super().__init__(model, config, cache_key, module_selector, seed=seed)
        self.select = select
        self.conv_percentage = conv_percentage
        self.linear_percentage = linear_percentage

    @torch.no_grad()
    def calibrate(self, *args: Any, **kwargs: Any) -> Dict[str, Dict[str, float]]:
        if self.conv_percentage == 0.0 and self.linear_percentage == 0.0:
            return {}
        if (
            (self.conv_percentage == 1.0 and self.linear_percentage == 1.0)
            or (self.conv_percentage == 0.0 and self.linear_percentage == 1.0)
            or (self.conv_percentage == 1.0 and self.linear_percentage == 0.0)
        ):
            # only_use_compute_density
            costs_calibrate_info = self.compute_quantization_costs(
                args, kwargs, module_selector=self.module_selector
            )
            costs_calibrate_info = self.apply_filter(costs_calibrate_info)
            if self.config.cache_dir is not None:
                self.save_quantization_status(
                    costs_calibrate_info, "quantization_stats.json"
                )
            return costs_calibrate_info

        calibrate_info = self.calibrate_all_layers(
            args, kwargs, module_selector=self.module_selector
        )

        selected_model = self.module_selector(self.model)

        # Initialize max and min values, as well as lists for linear and convolutional layer data
        max_value, min_value = 2, -1
        linear_values, conv_values = [max_value, min_value], [max_value, min_value]
        # Iterate through quantization information for each layer, extracting quantization values for linear and convolutional layers
        for module_name, value_info in calibrate_info.items():
            module = get_sub_module(selected_model, module_name)
            values_list = (
                linear_values if isinstance(module, nn.Linear) else conv_values
            )
            values_list.append(float(value_info[self.select]))

        # Sort quantization values for linear and convolutional layers based on the selected evaluation metric
        linear_values.sort()
        conv_values.sort()

        # Calculate linear and convolutional thresholds
        conv_threshold = conv_values[int((len(conv_values) - 1) * self.conv_percentage)]
        linear_threshold = linear_values[
            int((len(linear_values) - 1) * self.linear_percentage)
        ]
        # print(f"Conv threshold: {conv_threshold}, Linear threshold: {linear_threshold}")

        if nn.Conv2d in self.config.module_settings:
            self.config.module_settings[nn.Conv2d][self.select] = conv_threshold

        if nn.Linear in self.config.module_settings:
            self.config.module_settings[nn.Linear][self.select] = linear_threshold

        # Apply filters and save quantization status information
        calibrate_info = self.apply_filter(calibrate_info)
        if self.config.cache_dir is not None:
            self.save_quantization_status(calibrate_info, "quantization_stats.json")

        return calibrate_info


@dataclass
class OnelineQuantizationBoosterExecutor(BoosterExecutor):
    conv_percentage: int = 60
    linear_percentage: int = 70
    conv_compute_density_threshold: int = 100
    linear_compute_density_threshold: int = 300

    @singledispatchmethod
    def execute(self, model, ckpt_name=None, **kwargs):
        print(f"{type(self).__name__}.apply() not implemented for {type(model)}")
        return model

    @singledispatchmethod
    def extract_torch_module(self, model):
        raise NotImplementedError(f"{type(model)}")

    @execute.register(VAE)
    def _(self, model: VAE, **kwargs):
        # TODO: VAE does not support quantization and patch compatibility
        from .booster_basic import BasicOneFlowBoosterExecutor

        warnings.warn(
            "TODO: VAE does not support quantization and patch compatibility",
            UserWarning,
        )
        return BasicOneFlowBoosterExecutor().execute(model, **kwargs)

    @execute.register(ModelPatcher)
    @execute.register(ControlNet)
    def _(
        self, model: ModelPatcher, ckpt_name: Optional[str] = None, **kwargs
    ) -> Union[ModelPatcher, ControlNet]:
        # Define quantization settings
        quant_config = QuantizationConfig.from_settings(
            quantize_conv=True,
            quantize_linear=True,
            bits=8,
            conv_mae_threshold=0.9,
            linear_mae_threshold=0.9,
            plot_calibrate_info=True,
            conv_compute_density_threshold=self.conv_compute_density_threshold,
            linear_compute_density_threshold=self.linear_compute_density_threshold,
        )

        def get_torch_module(module):
            if isinstance(module, DeployableModule):
                return module._torch_module
            return module

        # Retrieve the model to be optimized
        def get_target_model(model, callback=None):
            if isinstance(model, ModelPatcher):
                target_model = model.model.diffusion_model
            elif isinstance(model, ControlNet):
                target_model = model.control_model
            else:
                raise NotImplementedError

            if callback:
                return callback(target_model)

            return target_model

        def set_optimized_model(model, quant_config):
            target_model = get_target_model(model)
            torch_model = get_torch_module(target_model)
            if isinstance(target_model, DeployableModule):
                compiled_model = target_model
            else:
                compiled_model = oneflow_compile(target_model)

            if isinstance(model, ModelPatcher):
                model.model.diffusion_model = compiled_model
            elif isinstance(model, ControlNet):
                model.control_model = compiled_model
            else:
                raise NotImplementedError

            compiled_model.apply_online_quant(quant_config)
            if ckpt_name:
                cache_key = f"{ckpt_name}_{type(torch_model).__name__}"
                graph_file = generate_graph_path(cache_key, torch_model)
                quant_config.cache_dir = os.path.dirname(graph_file)
                set_compiled_options(compiled_model, graph_file)

        set_environment_for_svd_img2vid(model)
        # Get the Torch model and set quantization calculator
        torch_model = get_target_model(model, callback=get_torch_module)
        if not is_fp16_model(torch_model):
            print(
                f"Warning: Model {type(torch_model)} is not an FP16 model. Compilation will be skipped!"
            )
            return model

        quant_config.quantization_calculator = SubQuantizationPercentileCalculator(
            torch_model,
            quant_config,
            cache_key=type(torch_model).__name__,
            conv_percentage=self.conv_percentage / 100,
            linear_percentage=self.linear_percentage / 100,
        )

        # Apply quantization optimization and return the optimized model
        if create_patch_executor(PatchType.DCUNetExecutorPatch).check_patch(model):
            self._set_optimized_model_for_deepcace(model)
        else:
            set_optimized_model(model, quant_config)
        return model

    def _set_optimized_model_for_deepcace(self, model: ModelPatcher):
        # TODO
        print(
            "Warning: DeepCache + OnelineQuantization only support default configurations:"
        )
        model.fast_deep_cache_unet.quantize = partial(
            quant_optimizer.quantize_model,
            model.fast_deep_cache_unet,
            quantize_conv=False,
        )
        model.deep_cache_unet.quantize = partial(
            quant_optimizer.quantize_model, model.deep_cache_unet, quantize_conv=False
        )
