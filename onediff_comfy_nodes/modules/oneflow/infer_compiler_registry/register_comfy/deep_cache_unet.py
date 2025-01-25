import torch
from comfy.ldm.modules.diffusionmodules.openaimodel import (
    apply_control,
    forward_timestep_embed,
)
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding
from torch.nn import Module


class DeepCacheUNet(Module):
    def __init__(self, unet_module, cache_layer_id, cache_block_id):
        super().__init__()
        self.unet_module = unet_module
        cache_depth = 0
        for i in range(cache_layer_id):
            cache_depth = cache_depth + self.unet_module.num_res_blocks[i]
        cache_depth = cache_depth + cache_block_id + 1
        self.cache_depth = cache_depth

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        c_dict = kwargs.get("c", {})
        num_video_frames = c_dict.get(
            "num_video_frames", self.unet_module.default_num_video_frames
        )

        default_image_only_indicator = getattr(
            self.unet_module, "default_image_only_indicator", None
        )
        image_only_indicator = c_dict.get(
            "image_only_indicator", default_image_only_indicator
        )

        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (
            self.unet_module.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(
            timesteps, self.unet_module.model_channels, repeat_only=False
        ).to(x.dtype)
        emb = self.unet_module.time_embed(t_emb)

        if self.unet_module.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.unet_module.label_emb(y)

        h = x
        cache_h = None
        for id, module in enumerate(self.unet_module.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(
                module,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            h = apply_control(h, control, "input")
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)

            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        if self.unet_module.middle_block is not None:
            h = forward_timestep_embed(
                self.unet_module.middle_block,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = apply_control(h, control, "middle")

        for id, module in enumerate(self.unet_module.output_blocks):
            if id == len(self.unet_module.output_blocks) - self.cache_depth - 1:
                cache_h = h
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, "output")

            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)

            h = torch.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1]
            else:
                output_shape = None
            h = forward_timestep_embed(
                module,
                h,
                emb,
                context,
                transformer_options,
                output_shape,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = h.type(x.dtype)
        if self.unet_module.predict_codebook_ids:
            return self.unet_module.id_predictor(h), cache_h
        else:
            return self.unet_module.out(h), cache_h


class FastDeepCacheUNet(Module):
    def __init__(self, unet_module, cache_layer_id, cache_block_id):
        super().__init__()
        self.unet_module = unet_module
        cache_depth = 0
        for i in range(cache_layer_id):
            cache_depth = cache_depth + self.unet_module.num_res_blocks[i]
        cache_depth = cache_depth + cache_block_id + 1
        self.cache_depth = cache_depth

    def forward(
        self,
        x,
        cache_h,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        c_dict = kwargs.get("c", {})
        num_video_frames = c_dict.get(
            "num_video_frames", self.unet_module.default_num_video_frames
        )

        default_image_only_indicator = getattr(
            self.unet_module, "default_image_only_indicator", None
        )
        image_only_indicator = c_dict.get(
            "image_only_indicator", default_image_only_indicator
        )

        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (
            self.unet_module.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(
            timesteps, self.unet_module.model_channels, repeat_only=False
        ).to(x.dtype)
        emb = self.unet_module.time_embed(t_emb)

        if self.unet_module.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.unet_module.label_emb(y)

        h = x
        for id, module in enumerate(self.unet_module.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(
                module,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            h = apply_control(h, control, "input")
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)

            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)
            if id == self.cache_depth:
                break

        for id, module in enumerate(self.unet_module.output_blocks):
            if id < len(self.unet_module.output_blocks) - self.cache_depth - 1:
                continue

            if id == len(self.unet_module.output_blocks) - self.cache_depth - 1:
                h = cache_h

            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, "output")

            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)

            h = torch.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1]
            else:
                output_shape = None
            h = forward_timestep_embed(
                module,
                h,
                emb,
                context,
                transformer_options,
                output_shape,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = h.type(x.dtype)
        if self.unet_module.predict_codebook_ids:
            return self.unet_module.id_predictor(h), cache_h
        else:
            return self.unet_module.out(h), cache_h
