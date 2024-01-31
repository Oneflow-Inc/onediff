from pathlib import Path
from typing import Optional, Union, Dict, Tuple
from collections import OrderedDict, defaultdict

import torch

from onediff.infer_compiler.utils.log_utils import logger
from onediff.infer_compiler.with_oneflow_compile import DualModule

from diffusers.loaders.lora import LoraLoaderMixin
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.utils import is_accelerate_available

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module


USE_PEFT_BACKEND = False


def offload_tensor(tensor, device):
    cur_device = tensor.device
    if cur_device == device:
        return tensor.clone()
    else:
        return tensor.to(device)


def linear_fuse_lora(
    self: torch.nn.Linear,
    state_dict: Dict[str, torch.Tensor],
    lora_scale: float = 1.0,
    alpha: float = None,
    rank: float = None,
    *,
    offload_device="cpu",
    offload_weight="lora",
):
    assert isinstance(self, torch.nn.Linear)
    if isinstance(self, DualModule):
        self = self._torch_module

    linear_unfuse_lora(self)
    dtype, device = self.weight.data.dtype, self.weight.data.device

    w_down = state_dict["lora.down.weight"].float().to(device)
    w_up = state_dict["lora.up.weight"].float().to(device)

    if alpha is not None:
        w_up = w_up * (alpha / rank * lora_scale)

    if offload_weight == "lora":
        self.register_buffer("_lora_up", offload_tensor(w_up, offload_device))
        self.register_buffer(
            "_lora_down", offload_tensor(state_dict["lora.down.weight"], offload_device)
        )
        self._lora_scale = lora_scale

    elif offload_weight == "weight":
        self.register_buffer(
            "_lora_orig_weight", offload_tensor(self.weight.data, offload_device)
        )

    else:
        raise ValueError(
            f"[OneDiff linear_fuse_lora] Invalid offload weight: {offload_weight}"
        )

    lora_weight = torch.bmm(w_up[None, :], w_down[None, :])[0]
    fused_weight = self.weight.data.float() + lora_weight
    self.weight.data.copy_(fused_weight.to(device=device, dtype=dtype))


def linear_unfuse_lora(self: torch.nn.Linear):
    assert isinstance(self, torch.nn.Linear)

    fused_weight = self.weight.data
    dtype, device = fused_weight.dtype, fused_weight.device

    if (
        "_lora_orig_weight" in self._buffers
        and self.get_buffer("_lora_orig_weight") is not None
    ):
        unfused_weight = self._lora_orig_weight
        self._lora_orig_weight = None

    elif "_lora_up" in self._buffers and self.get_buffer("_lora_up") is not None:
        w_up = self.get_buffer("_lora_up").to(device=device).float()
        w_down = self.get_buffer("_lora_down").to(device).float()

        unfused_weight = self.weight.data.float() - (
            torch.bmm(w_up[None, :], w_down[None, :])[0]
        )
        self._lora_up = None
        self._lora_down = None
        self._lora_scale = None

    else:
        return

    self.weight.data.copy_(unfused_weight.to(device=device, dtype=dtype))


def conv_fuse_lora(
    self: torch.nn.Conv2d,
    state_dict: Dict[str, torch.Tensor],
    lora_scale: float = 1.0,
    alpha: float = None,
    rank: float = None,
    *,
    offload_device="cpu",
    offload_weight="lora",
) -> None:
    assert isinstance(self, torch.nn.Conv2d)
    if isinstance(self, DualModule):
        self = self._torch_module
    conv_unfuse_lora(self)
    dtype, device = self.weight.data.dtype, self.weight.data.device

    w_down = state_dict["lora.down.weight"].float().to(device)
    w_up = state_dict["lora.up.weight"].float().to(device)

    if alpha is not None:
        w_up = w_up * (alpha / rank * lora_scale)

    if offload_weight == "lora":
        self.register_buffer("_lora_up", offload_tensor(w_up, offload_device))
        self.register_buffer(
            "_lora_down", offload_tensor(state_dict["lora.down.weight"], offload_device)
        )
        self._lora_scale = lora_scale
    elif offload_weight == "weight":
        self.register_buffer(
            "_lora_orig_weight", offload_tensor(self.weight.data, offload_device)
        )
    else:
        raise ValueError(
            f"[OneDiff conv_fuse_lora] Invalid offload weight: {offload_weight}"
        )

    lora_weight = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
    lora_weight = lora_weight.reshape((self.weight.shape))

    fused_weight = self.weight.data.float() + lora_weight
    self.weight.data.copy_(fused_weight.to(device=device, dtype=dtype))


def conv_unfuse_lora(self: torch.nn.Conv2d):
    assert isinstance(self, torch.nn.Conv2d)

    fused_weight = self.weight.data
    dtype, device = fused_weight.dtype, fused_weight.device

    if (
        "_lora_orig_weight" in self._buffers
        and self.get_buffer("_lora_orig_weight") is not None
    ):
        unfused_weight = self._lora_orig_weight
        self._lora_orig_weight = None

    elif "_lora_up" in self._buffers and self.get_buffer("_lora_up") is not None:
        w_up = self._lora_up.to(device=device).float()
        w_down = self._lora_down.to(device).float()

        fusion = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        fusion = fusion.reshape((fused_weight.shape))
        unfused_weight = fused_weight.float() - fusion

        self._lora_up = None
        self._lora_down = None
        self._lora_scale = None

    else:
        return

    self.weight.data.copy_(unfused_weight.to(device=device, dtype=dtype))

def load_and_fuse_lora(
    pipeline: LoraLoaderMixin,
    pretrained_model_name_or_path_or_dict: Union[str, Path, Dict[str, torch.Tensor]],
    adapter_name: Optional[str] = None,
    *,
    lora_scale: float = 1.0,
    offload_device="cpu",
    offload_weight="lora",
    use_cache=False,
    **kwargs,
) -> None:
    self = pipeline
    if adapter_name is not None:
        raise ValueError(
            f"[OneDiff load_and_fuse_lora] adapter_name != None is not supported"
        )

    if use_cache:
        state_dict, network_alphas = load_state_dict_cached(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
    else:
        state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )

    is_correct_format = all("lora" in key for key in state_dict.keys())
    if not is_correct_format:
        raise ValueError("[OneDiff load_and_fuse_lora] Invalid LoRA checkpoint.")

    # load lora into unet
    keys = list(state_dict.keys())
    cls = type(self)

    if all(
        key.startswith(cls.unet_name) or key.startswith(cls.text_encoder_name)
        for key in keys
    ):
        # Load the layers corresponding to UNet.
        logger.info(f"Loading {cls.unet_name}.")

        unet_keys = [k for k in keys if k.startswith(cls.unet_name)]
        state_dict = {
            k.replace(f"{cls.unet_name}.", ""): v
            for k, v in state_dict.items()
            if k in unet_keys
        }

        if network_alphas is not None:
            alpha_keys = [
                k for k in network_alphas.keys() if k.startswith(cls.unet_name)
            ]
            network_alphas = {
                k.replace(f"{cls.unet_name}.", ""): v
                for k, v in network_alphas.items()
                if k in alpha_keys
            }

        else:
            # Otherwise, we're dealing with the old format. This means the `state_dict` should only
            # contain the module names of the `unet` as its keys WITHOUT any prefix.
            warn_message = "You have saved the LoRA weights using the old format. To convert the old LoRA weights to the new format, you can first load them in a dictionary and then create a new dictionary like the following: `new_state_dict = {f'unet.{module_name}': params for module_name, params in old_state_dict.items()}`."
            logger.warn(warn_message)

    # unet.load_attn
    _pipeline = kwargs.pop("_pipeline", None)

    is_network_alphas_none = network_alphas is None

    is_lora = (
        all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys())
        and not USE_PEFT_BACKEND
    )
    is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())
    if is_custom_diffusion:
        raise ValueError(
            "[OneDiff load_and_fuse_lora] custom diffusion is not supported now."
        )

    if is_lora:
        # correct keys
        state_dict, network_alphas = self.unet.convert_state_dict_legacy_attn_format(
            state_dict, network_alphas
        )

        if network_alphas is not None:
            network_alphas_keys = list(network_alphas.keys())
            used_network_alphas_keys = set()

        lora_grouped_dict = defaultdict(dict)
        mapped_network_alphas = {}

        all_keys = list(state_dict.keys())
        for key in all_keys:
            value = state_dict.pop(key)
            attn_processor_key, sub_key = (
                ".".join(key.split(".")[:-3]),
                ".".join(key.split(".")[-3:]),
            )
            lora_grouped_dict[attn_processor_key][sub_key] = value

            # Create another `mapped_network_alphas` dictionary so that we can properly map them.
            if network_alphas is not None:
                for k in network_alphas_keys:
                    if k.replace(".alpha", "") in key:
                        mapped_network_alphas.update(
                            {attn_processor_key: network_alphas.get(k)}
                        )
                        used_network_alphas_keys.add(k)

        if not is_network_alphas_none:
            if len(set(network_alphas_keys) - used_network_alphas_keys) > 0:
                raise ValueError(
                    f"[OneDiff load_and_fuse_lora] The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
                )

        if len(state_dict) > 0:
            raise ValueError(
                f"[OneDiff load_and_fuse_lora] The `state_dict` has to be empty at this point but has the following keys \n\n {', '.join(state_dict.keys())}"
            )

        for key, value_dict in lora_grouped_dict.items():
            attn_processor = self.unet
            for sub_key in key.split("."):
                attn_processor = getattr(attn_processor, sub_key)

            # Process non-attention layers, which don't have to_{k,v,q,out_proj}_lora layers
            # or add_{k,v,q,out_proj}_proj_lora layers.
            rank = value_dict["lora.down.weight"].shape[0]

            if isinstance(attn_processor, LoRACompatibleConv):
                conv_fuse_lora(
                    attn_processor,
                    value_dict,
                    lora_scale,
                    mapped_network_alphas.get(key),
                    rank,
                    offload_device=offload_device,
                    offload_weight=offload_weight,
                )
            elif isinstance(attn_processor, LoRACompatibleLinear):
                linear_fuse_lora(
                    attn_processor,
                    value_dict,
                    lora_scale,
                    mapped_network_alphas.get(key),
                    rank,
                    offload_device=offload_device,
                    offload_weight=offload_weight,
                )
            else:
                raise ValueError(
                    f"[OneDiff load_and_fuse_lora] Module {key} is not a LoRACompatibleConv or LoRACompatibleLinear module."
                )
    else:
        raise ValueError(
            f"[OneDiff load_and_fuse_lora] {pretrained_model_name_or_path_or_dict} does not seem to be in the correct format expected by LoRA training."
        )

    is_model_cpu_offload = False
    is_sequential_cpu_offload = False

    if not USE_PEFT_BACKEND:
        if _pipeline is not None:
            for _, component in _pipeline.components.items():
                if isinstance(component, torch.nn.Module) and hasattr(
                    component, "_hf_hook"
                ):
                    is_model_cpu_offload = isinstance(
                        getattr(component, "_hf_hook"), CpuOffload
                    )
                    is_sequential_cpu_offload = isinstance(
                        getattr(component, "_hf_hook"), AlignDevicesHook
                    )

                    logger.info(
                        "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                    )
                    remove_hook_from_module(
                        component, recurse=is_sequential_cpu_offload
                    )

        # self.to(dtype=self.dtype, device=self.device)

        # Offload back.
        if is_model_cpu_offload:
            _pipeline.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            _pipeline.enable_sequential_cpu_offload()
        # Unsafe code />

    # load lora weights
    text_encoder_state_dict = {
        k: v for k, v in state_dict.items() if "text_encoder." in k
    }
    if len(text_encoder_state_dict) > 0:
        self.load_lora_into_text_encoder(
            text_encoder_state_dict,
            network_alphas=network_alphas,
            text_encoder=self.text_encoder,
            prefix="text_encoder",
            lora_scale=self.lora_scale,
            adapter_name=adapter_name,
            _pipeline=self,
        )

    text_encoder_2_state_dict = {
        k: v for k, v in state_dict.items() if "text_encoder_2." in k
    }
    if len(text_encoder_2_state_dict) > 0:
        self.load_lora_into_text_encoder(
            text_encoder_2_state_dict,
            network_alphas=network_alphas,
            text_encoder=self.text_encoder_2,
            prefix="text_encoder_2",
            lora_scale=self.lora_scale,
            adapter_name=adapter_name,
            _pipeline=self,
        )


def unfuse_lora(self: torch.nn.Module):
    def _unfuse_lora(m: torch.nn.Module):
        if isinstance(m, torch.nn.Linear):
            linear_unfuse_lora(m)
        elif isinstance(m, torch.nn.Conv2d):
            conv_unfuse_lora(m)

    self.apply(_unfuse_lora)


class LRUCacheDict(OrderedDict):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            oldest_key = next(iter(self))
            del self[oldest_key]
        super().__setitem__(key, value)


def load_state_dict_cached(
    lora: Union[str, Path, Dict[str, torch.Tensor]], **kwargs,
) -> Tuple[Dict, Dict]:
    assert isinstance(lora, (str, Path, dict))
    if isinstance(lora, dict):
        state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora, **kwargs)
        return state_dict, network_alphas

    global CachedLoRAs
    weight_name = kwargs.get("weight_name", None)

    lora_name = str(lora) + (f"/{weight_name}" if weight_name else "")
    if lora_name in CachedLoRAs:
        logger.debug(f"[OneDiff Cached LoRA] get cached lora of name: {str(lora_name)}")
        return CachedLoRAs[lora_name]

    state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora, **kwargs,)
    CachedLoRAs[lora_name] = (state_dict, network_alphas)
    logger.debug(f"[OneDiff Cached LoRA] create cached lora of name: {str(lora_name)}")
    return state_dict, network_alphas


CachedLoRAs = LRUCacheDict(100)
