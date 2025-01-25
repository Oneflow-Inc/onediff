import torch

import onediff_shared

from compile.utils import is_oneflow_backend

from onediff.infer_compiler import DeployableModule


class HijackLoraActivate:
    def __init__(self):
        from modules import extra_networks

        if "lora" in extra_networks.extra_network_registry:
            cls_extra_network_lora = type(extra_networks.extra_network_registry["lora"])
        else:
            cls_extra_network_lora = None
        self.lora_class = cls_extra_network_lora

    def __enter__(self):
        if self.lora_class is None:
            return
        self.orig_func = self.lora_class.activate
        self.lora_class.activate = hijacked_activate(self.lora_class.activate)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lora_class is None:
            return
        self.lora_class.activate = self.orig_func
        self.lora_class = None
        self.orig_func = None


def hijacked_activate(activate_func):
    import networks

    if hasattr(activate_func, "_onediff_hijacked"):
        return activate_func

    def activate(self, p, params_list):
        activate_func(self, p, params_list)
        sd_model = onediff_shared.current_unet_graph.graph_module
        if isinstance(sd_model, DeployableModule):
            onediff_sd_model: DeployableModule = sd_model
            for name, sub_module in onediff_sd_model.named_modules():
                if not isinstance(
                    sub_module,
                    (
                        torch.nn.Linear,
                        torch.nn.Conv2d,
                        torch.nn.GroupNorm,
                        torch.nn.LayerNorm,
                    ),
                ):
                    continue
                networks.network_apply_weights(sub_module)
                if is_oneflow_backend() and isinstance(sub_module, torch.nn.Conv2d):
                    # TODO(WangYi): refine here
                    from onediff.infer_compiler.backends.oneflow.param_utils import (
                        update_graph_related_tensor,
                    )

                    try:
                        update_graph_related_tensor(sub_module)
                    except:
                        pass

    activate._onediff_hijacked = True
    return activate
