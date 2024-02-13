# /home/fengwen/quant/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/sampling.py
from einops import rearrange
from oneflow.nn.functional import group_norm
import oneflow as flow
from onediff.infer_compiler.transform import register
from ._config import animatediff_pt, animatediff_hijacker, animatediff_of

FunctionInjectionHolder = animatediff_pt.animatediff.sampling.FunctionInjectionHolder
ADGS = animatediff_pt.animatediff.sampling.ADGS


def groupnorm_mm_factory(params):
    def groupnorm_mm_forward(self, input):
        # axes_factor normalizes batch based on total conds and unconds passed in batch;
        # the conds and unconds per batch can change based on VRAM optimizations that may kick in
        if not ADGS.is_using_sliding_context():
            axes_factor = input.size(0) // params.video_length
        else:
            axes_factor = input.size(0) // params.context_length

        # input = rearrange(input, "(b f) c h w -> b c f h w", b=axes_factor)
        # (b f) c h w -> b f c h w -> b c f h w
        input = input.unflatten(0, (axes_factor, -1)).permute(0, 2, 1, 3, 4)
        input = group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        # input = rearrange(input, "b c f h w -> (b f) c h w", b=axes_factor)
        # b c f h w -> b f c h w -> (b f) c h w
        input = input.permute(0, 2, 1, 3, 4).flatten(0, 1)
        return input

    return groupnorm_mm_forward


# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/utils_motion.py
GroupNormAD_OF_CLS = animatediff_of.animatediff.utils_motion.GroupNormAD
GroupNormAD_PT_CLS = animatediff_pt.animatediff.utils_motion.GroupNormAD
# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/motion_module_ad.py
AnimateDiffVersion = animatediff_pt.animatediff.motion_module_ad.AnimateDiffVersion
AnimateDiffFormat = animatediff_pt.animatediff.motion_module_ad.AnimateDiffFormat
# ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/animatediff/utils_model.py
ModelTypeSD = animatediff_pt.animatediff.utils_model.ModelTypeSD

_HANDLES = []


def inject_functions(orig_func, self, model, params):
    global _HANDLES

    ret = orig_func(self, model, params)
    # TODO  avoid call more than once
    info = model.motion_model.model.mm_info
    if not (
        info.mm_version == AnimateDiffVersion.V3
        or (
            info.mm_format == AnimateDiffFormat.ANIMATEDIFF
            and info.sd_type == ModelTypeSD.SD1_5
            and info.mm_version == AnimateDiffVersion.V2
            and params.apply_v2_models_properly
        )
    ):
        org_func = flow.nn.GroupNorm.forward
        flow.nn.GroupNorm.forward = groupnorm_mm_factory(params)

        def restore_groupnorm():
            flow.nn.GroupNorm.forward = org_func

        _HANDLES.append(restore_groupnorm)

        if params.apply_mm_groupnorm_hack:
            orig_func = GroupNormAD_OF_CLS.forward
            GroupNormAD_OF_CLS.forward = groupnorm_mm_factory(params)

            def restore_groupnorm_ad():
                GroupNormAD_OF_CLS.forward = orig_func

            _HANDLES.append(restore_groupnorm_ad)
    return ret


def restore_functions(orig_func, *args, **kwargs):
    global _HANDLES

    ret = orig_func(*args, **kwargs)
    for handle in _HANDLES:
        handle()
    _HANDLES = []
    return ret


def cond_func(*args, **kwargs):
    return True


animatediff_hijacker.register(
    FunctionInjectionHolder.inject_functions,
    inject_functions,
    cond_func,
)

animatediff_hijacker.register(
    FunctionInjectionHolder.restore_functions,
    restore_functions,
    cond_func,
)
