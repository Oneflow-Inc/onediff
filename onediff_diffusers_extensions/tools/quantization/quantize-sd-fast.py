import argparse
import time
from pathlib import Path

import torch

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download

from onediff.quantization import QuantPipeline
from PIL import Image
from safetensors.torch import load_file


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument("--quantized_model", type=str, required=True)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument(
    "--prompt",
    type=str,
    default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
)
parser.add_argument("--input_image", type=str, default=None)
parser.add_argument(
    "--conv_compute_density_threshold",
    type=int,
    default=900,
    help="The conv modules whose computational density is higher than the threshold will be quantized.",
)
parser.add_argument(
    "--linear_compute_density_threshold",
    type=int,
    default=300,
    help="The linear modules whose computational density is higher than the threshold will be quantized.",
)
parser.add_argument(
    "--conv_ssim_threshold",
    type=float,
    default=0,
    help="A similarity threshold that quantize convolution. The higher the threshold, the lower the accuracy loss caused by quantization.",
)
parser.add_argument(
    "--linear_ssim_threshold",
    type=float,
    default=0,
    help="A similarity threshold that quantize linear. The higher the threshold, the lower the accuracy loss caused by quantization.",
)
parser.add_argument(
    "--save_as_float",
    default=False,
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
)
parser.add_argument("--seed", type=int, default=111)
parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument(
    "--use_lightning",
    type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
    default=False,
    help="Use the SDXL Lightning model if true",
)
parser.add_argument(
    "--lightning_ckpt",
    type=str,
    default="sdxl_lightning_8step_unet.safetensors",
    help="Checkpoint file name for the ByteDance SDXL-Lightning model",
)
args = parser.parse_args()

pipeline_cls = (
    AutoPipelineForText2Image
    if args.input_image is None
    else AutoPipelineForImage2Image
)
is_safetensors_model = (
    Path(args.model).is_file and Path(args.model).suffix == ".safetensors"
)

if is_safetensors_model:
    try:  # check if safetensors is SDXL
        pipeline_cls = (
            StableDiffusionXLPipeline
            if args.input_image is None
            else StableDiffusionXLImg2ImgPipeline
        )
        pipe = QuantPipeline.from_single_file(
            pipeline_cls,
            args.model,
            torch_dtype=torch.float16,
            variant=args.variant,
            use_safetensors=True,
        )
    except:
        pipeline_cls = (
            StableDiffusionPipeline
            if args.input_image is None
            else StableDiffusionImg2ImgPipeline
        )
        pipe = QuantPipeline.from_single_file(
            pipeline_cls,
            args.model,
            torch_dtype=torch.float16,
            variant=args.variant,
            use_safetensors=True,
        )

if args.use_lightning:
    repo = "ByteDance/SDXL-Lightning"
    ckpt = args.lightning_ckpt
    unet = UNet2DConditionModel.from_config(args.model, subfolder="unet").to(
        "cuda", torch.float16
    )
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    pipe = QuantPipeline.from_pretrained(
        pipeline_cls,
        args.model,
        unet=unet,
        torch_dtype=torch.float16,
        variant=args.variant,
        use_safetensors=True,
    )
else:
    pipe = QuantPipeline.from_pretrained(
        pipeline_cls,
        args.model,
        torch_dtype=torch.float16,
        variant=args.variant,
        use_safetensors=True,
    )

pipe.to("cuda")

pipe_kwargs = dict(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.steps,
    seed=args.seed,
)
if args.input_image is not None:
    from diffusers.utils import load_image

    input_image = load_image(args.input_image)
    pipe_kwargs["image"] = input_image.resize((args.width, args.height), Image.LANCZOS)

start_time = time.time()

pipe.quantize(
    **pipe_kwargs,
    conv_compute_density_threshold=args.conv_compute_density_threshold,
    linear_compute_density_threshold=args.linear_compute_density_threshold,
    conv_ssim_threshold=args.conv_ssim_threshold,
    linear_ssim_threshold=args.linear_ssim_threshold,
    save_as_float=args.save_as_float,
    cache_dir=args.cache_dir,
)

pipe.save_quantized(args.quantized_model, safe_serialization=True)

end_time = time.time()

print(f"Quantize module time: {end_time - start_time}s")
