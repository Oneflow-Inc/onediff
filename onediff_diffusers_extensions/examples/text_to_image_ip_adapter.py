import argparse
import json
from pathlib import Path

import torch
from onediffx.compilers.diffusion_pipeline_compiler import (
    convert_pipe_to_memory_format,
)
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

nexfort_options = {
    "mode": "cudagraphs:benchmark:max-autotune:low-precision:cache-all",
    "memory_format": "channels_last",
    "options": {
        "inductor.optimize_linear_epilogue": False,
        "overrides.conv_benchmark": True,
        "overrides.matmul_allow_tf32": True,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="runwayml/stable-diffusion-v1-5"
)
parser.add_argument("--ipadapter", type=str, default="h94/IP-Adapter")
parser.add_argument("--subfolder", type=str, default="models")
parser.add_argument("--weight_name", type=str, default="ip-adapter_sd15.bin")
parser.add_argument("--scale", type=float, default=0.5)
parser.add_argument(
    "--input_image",
    type=str,
    default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png",
)
parser.add_argument(
    "--prompt", type=str, default="a cat",
)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--n_steps", type=int, default=30)
parser.add_argument(
    "--saved_image", type=str, required=False, default="ip-adapter-out.png"
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--run", type=int, default=3)
parser.add_argument(
    "--compile",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument(
    "--compile-ipa",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=True,
)
parser.add_argument("--compile-options", type=str, default=nexfort_options)
args = parser.parse_args()

# load an image
image = load_image(args.input_image)

# load stable diffusion and ip-adapter
pipe = AutoPipelineForText2Image.from_pretrained(
    args.base, torch_dtype=torch.float16
)
pipe.load_ip_adapter(
    args.ipadapter, subfolder=args.subfolder, weight_name=args.weight_name
)
pipe.set_ip_adapter_scale(args.scale)
pipe.to("cuda")


compile_options = args.compile_options
if isinstance(compile_options, str):
    compile_options = json.loads(compile_options)


memory_format = getattr(
    torch, compile_options["memory_format"], torch.channels_last
)
pipe = convert_pipe_to_memory_format(pipe, memory_format=memory_format)
compile_options.pop("memory_format", None)

if args.compile:
    from onediff.infer_compiler import compile

    pipe.unet = compile(pipe.unet, backend="nexfort", options=compile_options)
    pipe.vae.decoder = compile(
        pipe.vae.decoder, backend="nexfort", options=compile_options
    )
if args.compile_ipa:
    from onediff.infer_compiler import compile

    pipe.image_encoder = compile(
        pipe.image_encoder, backend="nexfort", options=compile_options
    )


# generate image
print("Warmup")
for i in range(args.warmup):
    images = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        ip_adapter_image=image,
        num_inference_steps=args.n_steps,
        generator=torch.manual_seed(args.seed),
    ).images

print("Run")
for i in range(args.run):
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        ip_adapter_image=image,
        num_inference_steps=args.n_steps,
        generator=torch.manual_seed(args.seed),
    ).images[0]
    image_path = f"{Path(args.saved_image).stem}_{i}" + Path(args.saved_image).suffix
    print(f"save output image to {image_path}")
    image.save(image_path)
