"""Convert torch object to oneflow object."""
import os
import importlib
import types
from functools import singledispatch
from collections import OrderedDict
from collections.abc import Iterable
from typing import Union, Any
import torch
import oneflow as flow

# TODO(strint): rm diffusers import
import diffusers

from .manager import transform_mgr, get_mock_cls_name
from ..import_tools import print_red, print_yellow


__all__ = [
    "proxy_class",
    "ProxySubmodule",
    "replace_obj",
    "replace_func",
    "map_args",
    "get_attr",
    "torch2oflow",
    "default_converter",
]


def proxy_class(cls: type):
    if cls.__module__.startswith("torch"):
        mod_name = cls.__module__.replace("torch", "oneflow")
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls.__name__)

    full_cls_name = get_mock_cls_name(cls)

    return transform_mgr.transform_cls(full_cls_name)


class ProxySubmodule:
    def __init__(self, submod):
        self._oflow_proxy_submod = submod
        self._oflow_proxy_parameters = {}
        self._oflow_proxy_children = {}

    def __getitem__(self, index):  # __getitem__
        if isinstance(self._oflow_proxy_submod, Iterable):
            submod = self._oflow_proxy_submod[index]
            return torch2oflow(submod)

        raise RuntimeError(f"can't getitem for: {type(self._oflow_proxy_submod)}")

    def __repr__(self) -> str:
        return " oflow_proxy: " + self._oflow_proxy_submod.__repr__()

    def __getattribute__(self, attribute):
        if attribute.startswith("_oflow_proxy"):
            return object.__getattribute__(self, attribute)
        elif attribute in ["forward", "_conv_forward"]:
            replacement = proxy_class(type(self._oflow_proxy_submod))
            return lambda *args, **kwargs: getattr(replacement, attribute)(
                self, *args, **kwargs
            )
        elif (
            isinstance(
                self._oflow_proxy_submod, diffusers.models.attention_processor.Attention
            )
            and attribute == "get_attention_scores"
        ):
            replacement = proxy_class(type(self._oflow_proxy_submod))
            return lambda *args, **kwargs: getattr(replacement, attribute)(
                self, *args, **kwargs
            )
        elif (
            isinstance(self._oflow_proxy_submod, torch.nn.Linear)
            and attribute == "use_fused_matmul_bias"
        ):
            return (
                self.bias is not None
                and os.getenv("ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR") == "1"
            )
        elif (
            isinstance(self._oflow_proxy_submod, torch.nn.Dropout)
            and attribute == "generator"
        ):
            return flow.Generator()
        elif (
            isinstance(self._oflow_proxy_submod, (torch.nn.Conv2d, torch.nn.Conv3d))
            and attribute == "channel_pos"
        ):
            return "channels_first"
        else:
            a = getattr(self._oflow_proxy_submod, attribute)

            if isinstance(a, (torch.nn.parameter.Parameter, torch.Tensor)):
                # TODO(oneflow): assert a.requires_grad == False
                if attribute not in self._oflow_proxy_parameters:
                    a = torch2oflow(a)

                    self._oflow_proxy_parameters[attribute] = a
                else:
                    a = self._oflow_proxy_parameters[attribute]
            elif isinstance(
                a, (torch.nn.Module, torch.nn.ModuleList, torch.nn.Sequential)
            ):
                if attribute not in self._oflow_proxy_children:
                    a = torch2oflow(a)

                    self._oflow_proxy_children[attribute] = a
                else:
                    a = self._oflow_proxy_children[attribute]

            return a

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        replacement = proxy_class(type(self._oflow_proxy_submod))

        if replacement is not None:
            return replacement.__call__(self, *args, **kwargs)
        else:
            raise RuntimeError(
                "can't find oneflow module for: " + str(type(self._oflow_proxy_submod))
            )


def replace_obj(obj):
    cls = type(obj)
    if cls == torch.dtype:
        return {
            "torch.float16": flow.float16,
            "torch.float32": flow.float32,
            "torch.double": flow.double,
            "torch.int8": flow.int8,
            "torch.int32": flow.int32,
            "torch.int64": flow.int64,
            "torch.uint8": flow.uint8,
        }[str(obj)]
    if cls == torch.fx.immutable_collections.immutable_list:
        return list(obj)
    replacement = proxy_class(cls)
    if replacement is not None:
        if cls in [torch.device]:
            return replacement(str(obj))
        elif cls == torch.nn.parameter.Parameter:
            return flow.utils.tensor.from_torch(obj.data)
        else:
            raise RuntimeError("don't know how to create oneflow obj for: " + str(cls))
    else:
        return obj


def replace_func(func):
    if func == torch.conv2d:
        return flow.nn.functional.conv2d
    if func == torch.conv3d:
        return flow.nn.functional.conv3d
    if func == torch._C._nn.linear:
        return flow.nn.functional.linear
    if func.__module__.startswith("torch"):
        mod_name = func.__module__.replace("torch", "oneflow")
        mod = importlib.import_module(mod_name)
        return getattr(mod, func.__name__)
    else:
        return func


def map_args(args, kwargs):
    args = [replace_obj(a) for a in args]
    kwargs = dict((k, replace_obj(v)) for (k, v) in kwargs.items())
    return (args, kwargs)


def get_attr(gm, node, torch2flow={}):
    attr = getattr(gm, node.target)
    if attr in torch2flow:
        return torch2flow[attr]
    of_attr = replace_obj(attr)
    torch2flow[attr] = of_attr
    return of_attr


@singledispatch
def torch2oflow(mod, *args, **kwargs):
    return default_converter(mod, *args, **kwargs)


def default_converter(obj, verbose=False, *, proxy_cls=None):
    try:
        new_obj_cls = proxy_class(type(obj)) if proxy_cls is None else proxy_cls

        def init(self):
            for k, _ in obj.__dict__.items():
                attr = getattr(obj, k)
                self.__dict__[k] = torch2oflow(attr)

        of_obj_cls = type(str(new_obj_cls), (new_obj_cls,), {"__init__": init})
        of_obj = of_obj_cls()

        if verbose:
            print(f"convert {type(obj)} to {type(of_obj)}")
        return of_obj
    except Exception as e:
        print_yellow(f"Unsupported type: {type(obj)}")
        return obj


@torch2oflow.register
def _(mod: torch.nn.Module, verbose=False):
    proxy_md = ProxySubmodule(mod)

    new_md_cls = proxy_class(type(mod))

    def init(self):
        nonlocal proxy_md

        # call the super `__init__` may cause unnecessary memory allocation,
        # so we call the nn.Module `__init__` instead.

        # super(type(self), self).__init__()
        flow.nn.Module.__init__(self)

        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        for n, p in list(proxy_md.named_parameters("", False)):
            self._parameters[n] = torch2oflow(p)
        for n, b in list(proxy_md.named_buffers("", False)):
            self._buffers[n] = flow.utils.tensor.from_torch(b.data)
        for n, m in proxy_md._modules.items():
            self._modules[n] = torch2oflow(m)

        for k, _ in proxy_md.__dict__.items():
            if k not in self.__dict__:
                attr = getattr(proxy_md, k)
                try:
                    self.__dict__[k] = torch2oflow(attr)

                except Exception as e:
                    print_red(f"convert {type(attr)} failed: {e}")
                    raise NotImplementedError(f"Unsupported type: {type(attr)}")

    def proxy_getattr(self, attr):
        nonlocal proxy_md

        try:
            return super().__getattribute__(attr)
        except:
            if attr in self._modules:
                return self._modules[attr]
            if attr in self._parameters:
                return self._parameters[attr]
            elif attr in self._buffers:
                return self._buffers[attr]
            else:
                return getattr(proxy_md, attr)

    of_mod_cls = type(
        str(new_md_cls), (new_md_cls,), {"__init__": init, "__getattr__": proxy_getattr}
    )
    of_mod = of_mod_cls()
    if of_mod.training:
        of_mod.training = False
        if verbose:
            print(
                f"""
            Warning: {type(of_mod)} is in training mode 
            and is turned into eval mode which is good for infrence optimation.
            """
            )

    if verbose:
        print(f"convert {type(mod)} to {type(of_mod)}")

    return of_mod


@torch2oflow.register
def _(mod: torch.nn.ModuleList, verbose=False):
    of_mod_list = flow.nn.ModuleList()
    for original_submod in mod:
        submod = torch2oflow(original_submod, verbose)
        of_mod_list.append(submod)

    return of_mod_list


@torch2oflow.register
def _(mod: torch.nn.Sequential, verbose=False):
    of_mod_list = []
    for original_submod in mod:
        submod = torch2oflow(original_submod, verbose)
        of_mod_list.append(submod)
    of_mod_seq = flow.nn.Sequential(*of_mod_list)

    return of_mod_seq


@torch2oflow.register
def _(mod: torch.nn.parameter.Parameter, verbose=False) -> flow.nn.Parameter:
    data = flow.utils.tensor.from_torch(mod.data)
    if mod.data.dtype == torch.int8:
        mod.requires_grad_(False)
        return flow.nn.Parameter(data.to(flow.int8), requires_grad=False)
    return flow.nn.Parameter(data, requires_grad=mod.requires_grad)


@torch2oflow.register
def _(mod: torch.Tensor, verbose=False) -> flow.Tensor:
    return flow.utils.tensor.from_torch(mod)


@torch2oflow.register
def _(mod: torch.dtype, verbose=False) -> flow.dtype:
    return {
        "torch.float16": flow.float16,
        "torch.float32": flow.float32,
        "torch.double": flow.double,
        "torch.int8": flow.int8,
        "torch.int32": flow.int32,
        "torch.int64": flow.int64,
        "torch.uint8": flow.uint8,
    }[str(mod)]


@torch2oflow.register
def _(mod: list, verbose=False) -> list:
    return [torch2oflow(m, verbose) for m in mod]


@torch2oflow.register
def _(mod: tuple, verbose=False) -> tuple:
    return tuple(torch2oflow(m, verbose) for m in mod)


@torch2oflow.register
def _(mod: OrderedDict, verbose=False) -> dict:
    return default_converter(mod, verbose, proxy_cls=OrderedDict)


@torch2oflow.register
def _(mod: set, verbose=False) -> set:
    return set(torch2oflow(m, verbose) for m in mod)


@torch2oflow.register(int)
@torch2oflow.register(float)
@torch2oflow.register(str)
@torch2oflow.register(bool)
def _(mod, verbose=False) -> Union[int, float, str, bool]:
    return mod


@torch2oflow.register
def _(mod: None, verbose=False):
    return mod


@torch2oflow.register
def _(mod: types.BuiltinFunctionType, verbose=False):
    if hasattr(mod, "__module__"):
        mod_name = None
        if mod.__module__.startswith("torch._C._nn"):
            mod_name = mod.__module__.replace(
                "torch._C._nn", "oneflow._oneflow_internal._C"
            )
        elif mod.__module__.startswith("torch"):
            try:
                if getattr(torch.nn.functional, mod.__name__) == mod:
                    mod_name = "oneflow.nn.functional"
            except:
                mod_name = mod.__module__.replace("torch", "oneflow")
        if mod_name is not None:
            m = importlib.import_module(mod_name)
            return getattr(m, mod.__name__)

    return default_converter(mod, verbose)


@torch2oflow.register
def _(mod: torch.device, verbose=False):
    index = mod.index if mod.index is not None else 0
    return flow.device(mod.type, index)


try:
    from onediff.optimization.attention_processor import FusedSelfAttnProcessor

    @torch2oflow.register
    def _(mod: FusedSelfAttnProcessor, verbose=False) -> FusedSelfAttnProcessor:
        return mod


except:
    pass
