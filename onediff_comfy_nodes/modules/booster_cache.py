from collections import OrderedDict
from functools import singledispatch

import torch
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
from onediff.torch_utils.module_operations import get_sub_module


@singledispatch
def switch_to_cached_model(new_model, cached_model):
    raise NotImplementedError(type(new_model))


@switch_to_cached_model.register
def _(new_model: ModelPatcher, cached_model):
    if type(new_model.model) != type(cached_model):
        raise TypeError(
            f"Model type mismatch: expected {type(cached_model)}, got {type(new_model.model)}"
        )

    cached_model.diffusion_model.load_state_dict(
        new_model.model.diffusion_model.state_dict(), strict=True
    )
    new_model.model.diffusion_model = cached_model.diffusion_model
    new_model.weight_inplace_update = True
    return new_model


@switch_to_cached_model.register
def _(new_model: VAE, cached_model):
    assert type(new_model.first_stage_model) == type(cached_model)
    for k, v in new_model.first_stage_model.state_dict().items():
        cached_v: torch.Tensor = get_sub_module(cached_model, k)
        assert v.dtype == cached_v.dtype
        cached_v.copy_(v)
    new_model.first_stage_model = cached_model
    return new_model


@singledispatch
def get_cached_model(model):
    return None
    # raise NotImplementedError(type(model))


@get_cached_model.register
def _(model: ModelPatcher):
    return model.model


@get_cached_model.register
def _(model: VAE):
    # from onediff.utils.import_utils import is_oneflow_available
    # from .._config import is_disable_oneflow_backend
    # if is_oneflow_available() and not is_disable_oneflow_backend():
    #     from .oneflow.utils.booster_utils import is_using_oneflow_backend

    #     if is_using_oneflow_backend(model):
    #         return None

    # # TODO(TEST) if support cache
    # # return model.first_stage_model
    return None


class BoosterCacheService:
    _cache = OrderedDict()

    def put(self, key, model):
        if key is None:
            return
        # oneflow backends output image error
        cached_model = get_cached_model(model)
        if cached_model:
            self._cache[key] = cached_model

    def get(self, key, default=None):
        return self._cache.get(key, default)

    def get_cached_model(self, key, model):
        cached_model = self.get(key, None)
        print(f"Cache lookup: Key='{key}', Cached Model Type='{type(cached_model)}'")
        if cached_model is not None:
            try:
                return switch_to_cached_model(model, cached_model)
            except Exception as e:
                print(f"An exception occurred when switching to cached model:")
                del self._cache[key]
                torch.cuda.empty_cache()

        return None
