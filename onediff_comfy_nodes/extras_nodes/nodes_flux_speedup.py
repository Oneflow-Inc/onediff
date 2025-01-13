import comfy
from comfy.model_patcher import ModelPatcher
import folder_paths


from comfy.ldm.flux.layers import RMSNorm as SrcRMSNorm
from nexfort.nn.modules import RMSNorm4Comfy4D

from comfy.ldm.flux.layers import MLPEmbedder as SrcMLPEmbedder
from nexfort.nn.modules import MLPEmbedder

from comfy.ldm.flux.layers import Modulation as SrcModulation
from nexfort.nn.modules import Modulation

from comfy.ldm.flux.layers import SingleStreamBlock as SrcSingleStreamBlock
from nexfort.nn.modules import SingleStreamBlock

from comfy.ldm.flux.layers import DoubleStreamBlock as SrcDoubleStreamBlock
from nexfort.nn.modules import DoubleStreamBlock

from comfy.ldm.flux.layers import LastLayer as SrcLastLayer
from nexfort.nn.modules import LastLayer



from contextlib import contextmanager
from comfy.model_management import VRAMState

@contextmanager
def comfy_cpu_context():
    orig_state = comfy.model_management.vram_state
    comfy.model_management.vram_state = VRAMState.NORMAL_VRAM
    try:
        yield
    finally:
        comfy.model_management.vram_state = orig_state

comfy2nexfort_class_map = {
    SrcRMSNorm: RMSNorm4Comfy4D,
    SrcMLPEmbedder: MLPEmbedder,
    SrcModulation: Modulation,
    SrcSingleStreamBlock: SingleStreamBlock,
    SrcDoubleStreamBlock: DoubleStreamBlock,
    SrcLastLayer: LastLayer,
}

from nexfort.compilers.custom_transform import transform_mgr

transform_mgr.update_class_proxies(comfy2nexfort_class_map)


from nexfort.compilers.transform_model import transform_model
from nexfort.quantization import quantize


def speedup(model: ModelPatcher, enable_quantize: bool, enable_transform: bool):
    import os

    os.environ["CUBLASLT_WORKSPACE_SIZE"] = str(1024)

    if enable_quantize:
        quantize(
            model.model.diffusion_model,
            quant_type="fp8_e4m3_e4m3_dynamic_per_tensor",
        )
    if enable_transform:
        transform_model(model.model.diffusion_model)


class OneDiffFluxSpeedup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "quantize_model": (
                    "BOOLEAN",
                    {"default": True, "label_on": "yes", "label_off": "no"},
                ),
                "transform": (
                    "BOOLEAN",
                    {"default": True, "label_on": "yes", "label_off": "no"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "speedup"
    CATEGORY = "OneDiffEnterprise"

    def speedup(self, model, quantize_model=True, transform=True):
        speedup(model, quantize_model, transform)
        return (model,)


class OneDiffFluxLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "enable_quant": (
                    "BOOLEAN",
                    {"default": True, "label_on": "yes", "label_off": "no"},
                ),
                "enable_transform": (
                    "BOOLEAN",
                    {"default": True, "label_on": "yes", "label_off": "no"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "OneDiffEnterprise"

    def load_unet(self, unet_name, enable_quant, enable_transform):
        model_options = {}
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        with comfy_cpu_context():
            model = comfy.sd.load_diffusion_model(
                unet_path, model_options=model_options
            )
        model.offload_device = comfy.model_management.unet_offload_device()
        speedup(model, enable_quant, enable_transform)

        return (model,)


NODE_CLASS_MAPPINGS = {
    "OneDiffFluxSpeedup": OneDiffFluxSpeedup,
    "OneDiffFluxLoader": OneDiffFluxLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OneDiffFluxSpeedup": "OneDiff Flux Speedup - OneDiffEnterprise",
    "OneDiffFluxLoader": "OneDiff Flux Speedup Loader - OneDiffEnterprise",
}
