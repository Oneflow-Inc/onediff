"""
example: python examples/text_to_image.py --height 512 --width 512 --warmup 10

Can be used with torch==2.4.0 diffusers==0.29.2
"""
import argparse

import torch
import oneflow as flow  # usort: skip

import functools

import torch._dynamo
from diffusers import StableDiffusionPipeline

from nexfort.frontends.diffusers.diffusion_pipeline_compiler import compile_pipe
from onediff.infer_compiler import oneflow_compile
from onediff.schedulers import EulerDiscreteScheduler

torch._dynamo.config.suppress_errors = True
TORCHDYNAMO_VERBOSE = 1

compiler_ignores = []
COMPILER_CONFIG = '{"mode": "max-autotune:benchmark:low-precision"}'
# COMPILER_CONFIG = '{"mode": "low-precision"}'
# COMPILER_CONFIG = '{"mode": "max-autotune"}'
MEMORY_FORMAT = "channels_last"
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo of image generation.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of an astronaut riding a horse on mars with long hair",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--compiler", type=str, default="nexfort")
    parser.add_argument("--compiler-config", type=str, default=COMPILER_CONFIG)
    parser.add_argument("--fuse-qkv-projections", action="store_true")
    parser.add_argument("--memory-format", type=str, default=MEMORY_FORMAT)

    cmd_args = parser.parse_args()
    return cmd_args


args = parse_args()

scheduler = EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    args.model_id,
    scheduler=scheduler,
    use_auth_token=True,
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe = pipe.to("cuda")
pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
memory_format = getattr(torch, args.memory_format)
compiler_config = (
    json.loads(args.compiler_config) if args.compiler_config is not None else {}
)

if args.compiler in ("none", "nexfort"):
    if args.compiler == "none":
        compiler_config["disable"] = True
    compile_pipe_ = functools.partial(
        compile_pipe,
        ignores=compiler_ignores,
        config=compiler_config,
        fuse_qkv_projections=args.fuse_qkv_projections,
        memory_format=memory_format,
    )
    prior_pipe = compile_pipe_(pipe)

elif args.compiler == "inductor":
    compile_pipe_ = functools.partial(
        compile_pipe,
        ignores=compiler_ignores,
        config={
            "backend": "inductor",
            **inductor_config,
        },
        fuse_qkv_projections=args.fuse_qkv_projections,
        memory_format=memory_format,
    )
    prior_pipe = compile_pipe_(prior_pipe)
    decoder_pipe = compile_pipe_(decoder_pipe)

prompt = args.prompt
with flow.autocast("cuda"):
    for _ in range(args.warmup):
        torch.manual_seed(args.seed)
        images = pipe(
            prompt, height=args.height, width=args.width, num_inference_steps=args.steps
        ).images

    torch.manual_seed(args.seed)
    images = pipe(
        prompt, height=args.height, width=args.width, num_inference_steps=args.steps
    ).images
    for i, image in enumerate(images):
        image.save(f"{prompt}-of-{i}.png")
