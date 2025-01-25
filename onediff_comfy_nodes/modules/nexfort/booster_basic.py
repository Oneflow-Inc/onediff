from functools import partial, singledispatchmethod
from typing import Optional

import torch

from comfy.controlnet import ControlLora, ControlNet
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
from nexfort.utils.memory_format import apply_memory_format

from onediff.infer_compiler import compile

from ..booster_interface import BoosterExecutor
from .onediff_controlnet import OneDiffControlLora


class BasicNexFortBoosterExecutor(BoosterExecutor):
    # https://pytorch.org/docs/stable/_modules/torch.html#compile
    def __init__(
        self,
        mode: str = "max-optimize:max-autotune:low-precision",
        option: str ="",
        fullgraph=False,
        dynamic=True,
    ):
        super().__init__()
        if isinstance(option, str):
            import json
            option = json.loads(option)
        options = {
            "mode": mode,
            "options": option,
            "dynamic": dynamic,
            "fullgraph": fullgraph,
        }
        self.compile_fn = partial(compile, backend="nexfort", options=options)
        self.options = options

    @singledispatchmethod
    def execute(self, model, ckpt_name=None, **kwargs):
        raise NotImplementedError(f"Cannot execute {type(model)=}")

    @execute.register(ModelPatcher)
    @torch.inference_mode()
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        diffusion_model = model.model.diffusion_model
        model.model.diffusion_model = apply_memory_format(
            diffusion_model, torch.channels_last
        )
        model.model.diffusion_model = self.compile_fn(diffusion_model)
        model.weight_inplace_update = True
        return model

    @execute.register(VAE)
    @torch.inference_mode()
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        model.first_stage_model = apply_memory_format(
            model.first_stage_model, torch.channels_last
        )
        print(f"{type(model)} apply compiled config: {self.options}")
        # https://huggingface.co/blog/sd3#performance-optimizations-for-sd3
        model.first_stage_model.decode = self.compile_fn(model.first_stage_model.decode)
        return model

    @execute.register(ControlNet)
    @torch.inference_mode()
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        torch_model = model.control_model
        torch_model = apply_memory_format(torch_model, torch.channels_last)
        compiled_model: torch.nn.Module = self.compile_fn(torch_model)
        model.control_model = compiled_model
        return model

    @execute.register(ControlLora)
    @torch.inference_mode()
    def _(self, model, ckpt_name: Optional[str] = None, **kwargs):
        def compile_cnet(model):
            out: torch.nn.Module = self.compile_fn(model)
            return out

        model = OneDiffControlLora.from_controllora(model, compile_fn=compile_cnet)
        return model
