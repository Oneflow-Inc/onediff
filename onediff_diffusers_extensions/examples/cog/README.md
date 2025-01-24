RUN:

export NEXFORT_FX_FORCE_TRITON_SDPA=1

python3 onediff_diffusers_extensions/examples/cog/text_to_image_cog.py --compiler nexfort --compiler-config '{"mode": "max-optimize:max-autotune:max-autotune", "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": false, "triton.fuse_attention_allow_fp16_reduction": false}}'

python3 onediff_diffusers_extensions/examples/cog/text_to_image_cog.py --compiler torch

python3 onediff_diffusers_extensions/examples/cog/text_to_image_cog.py --compiler none
