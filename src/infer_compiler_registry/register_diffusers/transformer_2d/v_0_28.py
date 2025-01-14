from typing import Any, Dict, Optional

import oneflow as torch

from onediff.infer_compiler.backends.oneflow.transform import transform_mgr

transformed_diffusers = transform_mgr.transform_package("diffusers")
ConfigMixin = transformed_diffusers.configuration_utils.ConfigMixin
register_to_config = transformed_diffusers.configuration_utils.register_to_config
ImagePositionalEmbeddings = (
    transformed_diffusers.models.embeddings.ImagePositionalEmbeddings
)
USE_PEFT_BACKEND = transformed_diffusers.utils.USE_PEFT_BACKEND
BaseOutput = transformed_diffusers.utils.BaseOutput
deprecate = transformed_diffusers.utils.deprecate
is_torch_version = transformed_diffusers.utils.is_torch_version
BasicTransformerBlock = transformed_diffusers.models.attention.BasicTransformerBlock
PatchEmbed = transformed_diffusers.models.embeddings.PatchEmbed
PixArtAlphaTextProjection = (
    transformed_diffusers.models.embeddings.PixArtAlphaTextProjection
)
LoRACompatibleConv = transformed_diffusers.models.lora.LoRACompatibleConv
LoRACompatibleLinear = transformed_diffusers.models.lora.LoRACompatibleLinear
ModelMixin = transformed_diffusers.models.modeling_utils.ModelMixin
AdaLayerNormSingle = transformed_diffusers.models.normalization.AdaLayerNormSingle
Transformer2DModelOutput = (
    transformed_diffusers.models.transformers.transformer_2d.Transformer2DModelOutput
)
proxy_Transformer2DModel = (
    transformed_diffusers.models.transformers.transformer_2d.Transformer2DModel
)


class Transformer2DModel(proxy_Transformer2DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # import ipdb; ipdb.set_trace()
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.Tensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformers.transformer_2d.Transformer2DModelOutput`] is returned,
            otherwise a `tuple` where the first element is the sample tensor.
        """
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored."
                )
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        hidden_states_in = hidden_states

        # 1. Input
        if self.is_input_continuous:
            batch_size, _, height, width = hidden_states.shape
            residual = hidden_states
            hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            height, width = (
                hidden_states.shape[-2] // self.patch_size,
                hidden_states.shape[-1] // self.patch_size,
            )
            (
                hidden_states,
                encoder_hidden_states,
                timestep,
                embedded_timestep,
            ) = self._operate_on_patched_inputs(
                hidden_states, encoder_hidden_states, timestep, added_cond_kwargs
            )

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = (
                    hidden_states.permute(0, 2, 1)
                    .reshape_as(hidden_states_in)
                    .contiguous()
                )
                # hidden_states = (
                #     hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                # )
                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = (
                    hidden_states.permute(0, 2, 1)
                    .reshape_as(hidden_states_in)
                    .contiguous()
                )
                # hidden_states = (
                #     hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                # )

            output = hidden_states + residual
            # output = self._get_output_for_continuous_inputs(
            #     hidden_states=hidden_states,
            #     residual=residual,
            #     batch_size=batch_size,
            #     height=height,
            #     width=width,
            #     inner_dim=inner_dim,
            # )
        elif self.is_input_vectorized:
            output = self._get_output_for_vectorized_inputs(hidden_states)
        elif self.is_input_patches:
            output = self._get_output_for_patched_inputs(
                hidden_states=hidden_states,
                timestep=timestep,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep,
                height=height,
                width=width,
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _get_output_for_continuous_inputs(
        self, hidden_states, residual, batch_size, height, width, inner_dim
    ):
        # # hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        # hidden_states = (
        #     hidden_states.permute(0, 2, 1)
        #     .reshape_as(hidden_states_in)
        #     .contiguous()
        # )
        # hidden_states = self.proj_out(hidden_states)
        # import ipdb; ipdb.set_trace()
        return
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch_size, height, width, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch_size, height, width, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        output = hidden_states + residual
        return output

    def _get_output_for_vectorized_inputs(self, hidden_states):
        raise
        hidden_states = self.norm_out(hidden_states)
        logits = self.out(hidden_states)
        # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
        logits = logits.permute(0, 2, 1)
        # log(p(x_0))
        output = F.log_softmax(logits.double(), dim=1).float()
        return output

    def _get_output_for_patched_inputs(
        self,
        hidden_states,
        timestep,
        class_labels,
        embedded_timestep,
        height=None,
        width=None,
    ):
        raise
        # import ipdb; ipdb.set_trace()
        if self.config.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = (
                self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            )
            hidden_states = self.proj_out_2(hidden_states)
        elif self.config.norm_type == "ada_norm_single":
            shift, scale = (
                self.scale_shift_table[None] + embedded_timestep[:, None]
            ).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                height,
                width,
                self.patch_size,
                self.patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                -1,
                self.out_channels,
                height * self.patch_size,
                width * self.patch_size,
            )
        )
        return output

    def _operate_on_continuous_inputs(self, hidden_states):
        batch, _, height, width = hidden_states.shape
        hidden_states = self.norm(hidden_states)

        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            # hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = hidden_states.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            inner_dim = hidden_states.shape[1]
            # hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = hidden_states.permute(0, 2, 3, 1).flatten(1, 2)
            hidden_states = self.proj_in(hidden_states)

        return hidden_states, inner_dim
