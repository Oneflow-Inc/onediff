import collections
import os

import yaml
from onediff.infer_compiler.backends.nexfort import fallback_to_eager

from ..modules.nexfort.booster_basic import BasicNexFortBoosterExecutor


# https://github.com/siliconflow/nexfort?tab=readme-ov-file#suggested-combinations-of-compiler-modes
compiler_modes = collections.OrderedDict(
    {
        "max-optimize:max-autotune:low-precision": "This will deliver a good performance and adapt quickly to shape changes.",
        "max-optimize:max-autotune:low-precision:freezing:benchmark": "",
        "jit:disable-runtime-fusion:low-precision": "This compiles super quickly, but the performance might not be optimized very noticeably.",
        "jit:benchmark:low-precision:freezing:cudagraphs": "This compiles the model very quickly, but the performance might be not as good as `TorchInductor` optimized models.",
        "max-autotune:benchmark:low-precision:cudagraphs": "This is the most suggested combination of compiler modes. It will deliver a good balance between performance and compilation time.",
        "max-optimize:max-autotune:benchmark:low-precision:freezing:cudagraphs": "This is the most aggressive combination of compiler modes. It will deliver the best performance but might slow down the compilation significantly.",
    }
)
options = collections.OrderedDict(
    {
        '{"inductor.optimize_linear_epilogue": false}': "",
        '{"triton.fuse_attention_allow_fp16_reduction": false, "inductor.optimize_linear_epilogue": false}': "",
    }
)


class OneDiffNexfortBooster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fullgraph": ([False, True],),
                "dynamic": ([True, False],),
                "mode": ([mode for mode in compiler_modes.keys()],),
                "option": ([option for option in options.keys()],),
                "docs_link": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[Note]: \nInstall-nexfort \nhttps://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort#install-nexfort",
                    },
                ),
            }
        }

    CATEGORY = "OneDiff/Booster"
    RETURN_TYPES = ("TorchCompileBooster",)
    FUNCTION = "apply"

    def apply(
        self,
        fullgraph=False,
        dynamic=None,
        mode="max-autotune:cudagraphs",
        option="",
        docs_link=None,
    ):
        return (
            BasicNexFortBoosterExecutor(
                fullgraph=fullgraph, mode=f"{mode}:cache-all", option=option, dynamic=dynamic
            ),
        )


class ResolutionSpeedupChecker:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "resolutions_config.yaml")
    with open(config_path, "r") as file:
        resolutions = yaml.safe_load(file)
    height_width_dict = {x["height"]: x["width"] for x in resolutions["resolutions"]}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "check"

    def check(self, samples):
        _, _, H, W = samples["samples"].shape
        H, W = H * 8, W * 8
        if H in self.height_width_dict and self.height_width_dict[H] == W:
            fallback_to_eager(True)
        else:
            fallback_to_eager(False)
        return (samples,)


NODE_CLASS_MAPPINGS = {
    "OneDiffNexfortBooster": OneDiffNexfortBooster,
    "ResolutionSpeedupChecker": ResolutionSpeedupChecker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OneDiffNexfortBooster": "Nexfort Booster - OneDiff",
    "ResolutionSpeedupChecker": "Speedup Checker - Resolution",
}
