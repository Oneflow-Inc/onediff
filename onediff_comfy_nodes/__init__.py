"""OneDiff ComfyUI Speedup Module"""
from onediff.utils.import_utils import is_nexfort_available, is_oneflow_available

from ._config import is_disable_oneflow_backend
from ._nodes import (
    ControlnetSpeedup,
    ModelSpeedup,
    OneDiffApplyModelBooster,
    OneDiffCheckpointLoaderSimple,
    OneDiffControlNetLoader,
    VaeSpeedup,
)

NODE_CLASS_MAPPINGS = {
    "ModelSpeedup": ModelSpeedup,
    "VaeSpeedup": VaeSpeedup,
    "ControlnetSpeedup": ControlnetSpeedup,
    "OneDiffModelBooster": OneDiffApplyModelBooster,
    "OneDiffCheckpointLoaderSimple": OneDiffCheckpointLoaderSimple,
    "OneDiffControlNetLoader": OneDiffControlNetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelSpeedup": "Model Speedup",
    "VaeSpeedup": "VAE Speedup",
    "OneDiffModelBooster": "Apply Model Booster - OneDiff",
    "ControlnetSpeedup": "ControlNet Speedup",
    "OneDiffCheckpointLoaderSimple": "Load Checkpoint - OneDiff",
}


def update_node_mappings(node):
    NODE_CLASS_MAPPINGS.update(node.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(node.NODE_DISPLAY_NAME_MAPPINGS)


def lazy_load_extra_nodes():

    from .extras_nodes import nodes_torch_compile_booster

    update_node_mappings(nodes_torch_compile_booster)

    if is_oneflow_available() and not is_disable_oneflow_backend():
        from .extras_nodes import nodes_compare, nodes_oneflow_booster

        update_node_mappings(nodes_oneflow_booster)
        update_node_mappings(nodes_compare)

    if is_nexfort_available():
        from .extras_nodes import nodes_nexfort_booster, nodes_flux_speedup

        update_node_mappings(nodes_nexfort_booster)
        update_node_mappings(nodes_flux_speedup)

    from .extras_nodes import nodes_prompt_styler

    update_node_mappings(nodes_prompt_styler)


# Lazy load all extra nodes when needed
lazy_load_extra_nodes()
