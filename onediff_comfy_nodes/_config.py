import os
import sys

import folder_paths
import torch._dynamo
torch._dynamo.config.suppress_errors = True

__all__ = [
    "is_default_using_oneflow_backend",
    "is_default_using_nexfort_backend",
    "is_disable_oneflow_backend",
]

# https://github.com/comfyanonymous/ComfyUI/blob/master/folder_paths.py#L9
os.environ["COMFYUI_ROOT"] = folder_paths.base_path
_default_backend = os.environ.get("ONEDIFF_COMFY_NODES_DEFAULT_BACKEND", "oneflow")
_disable_oneflow_backend = (
    os.environ.get("ONEDIFF_COMFY_NODES_DISABLE_ONEFLOW_BACKEND", "0") == "1"
)

custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes")

# Add paths to sys.path if not already there
if custom_nodes_path not in sys.path:
    sys.path.append(custom_nodes_path)


if _default_backend not in ["oneflow", "nexfort"]:
    raise ValueError(f"Invalid default backend: {_default_backend}")


def is_disable_oneflow_backend():
    return _disable_oneflow_backend


def is_default_using_nexfort_backend():
    return _default_backend == "nexfort"


def is_default_using_oneflow_backend():
    return _default_backend == "oneflow" and not _disable_oneflow_backend
