import argparse
import json
import os
import time

import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from onediffx import compile_pipe, load_pipe, quantize_pipe, save_pipe
from onediffx.utils.performance_monitor import track_inference_time
from safetensors.torch import load_file

try:
    USE_PEFT_BACKEND = diffusers.utils.USE_PEFT_BACKEND
except Exception as e:
    USE_PEFT_BACKEND = False

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--repo", type=str, default="ByteDance/SDXL-Lightning")
parser.add_argument("--cpkt", type=str, default="sdxl_lightning_8step_unet.safetensors")
parser.add_argument("--variant", type=str, default="fp16")
parser.add_argument(
    "--prompt",
    type=str,
    # default="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
    default="A girl smiling",
)
parser.add_argument("--save_graph", action="store_true")
parser.add_argument("--load_graph", action="store_true")
parser.add_argument("--save_graph_dir", type=str, default="cached_pipe")
parser.add_argument("--load_graph_dir", type=str, default="cached_pipe")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument(
    "--saved_image", type=str, required=False, default="sdxl-light-out.png"
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--compiler",
    type=str,
    default="oneflow",
    help="Compiler backend to use. Options: 'none', 'nexfort', 'oneflow'",
)
parser.add_argument(
    "--compiler-config", type=str, help="JSON string for nexfort compiler config."
)
parser.add_argument(
    "--quantize-config", type=str, help="JSON string for nexfort quantization config."
)
parser.add_argument("--bits", type=int, default=8)
parser.add_argument("--use_quantization", action="store_true")


args = parser.parse_args()

OUTPUT_TYPE = "pil"

n_steps = int(args.cpkt[len("sdxl_lightning_") : len("sdxl_lightning_") + 1])

is_lora_cpkt = "lora" in args.cpkt

if args.compiler == "oneflow":
    from onediff.schedulers import EulerDiscreteScheduler
else:
    from diffusers import EulerDiscreteScheduler

if is_lora_cpkt:
    if not USE_PEFT_BACKEND:
        print("PEFT backend is required for load_lora_weights")
        exit(0)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    if os.path.isfile(os.path.join(args.repo, args.cpkt)):
        pipe.load_lora_weights(os.path.join(args.repo, args.cpkt))
    else:
        pipe.load_lora_weights(hf_hub_download(args.repo, args.cpkt))
    pipe.fuse_lora()
else:
    if args.use_quantization and args.compiler == "oneflow":
        print("oneflow backend quant...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.base, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        import onediff_quant
        from onediff_quant.utils import replace_sub_module_with_quantizable_module

        quantized_layers_count = 0
        onediff_quant.enable_load_quantized_model()

        calibrate_info = {}
        with open(os.path.join(args.base, "calibrate_info.txt"), "r") as f:
            for line in f.readlines():
                line = line.strip()
                items = line.split(" ")
                calibrate_info[items[0]] = [
                    float(items[1]),
                    int(items[2]),
                    [float(x) for x in items[3].split(",")],
                ]

        for sub_module_name, sub_calibrate_info in calibrate_info.items():
            replace_sub_module_with_quantizable_module(
                pipe.unet,
                sub_module_name,
                sub_calibrate_info,
                False,
                False,
                args.bits,
            )
            quantized_layers_count += 1

        print(f"Total quantized layers: {quantized_layers_count}")

    else:
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_config(args.base, subfolder="unet").to(
            "cuda", torch.float16
        )
        if os.path.isfile(os.path.join(args.repo, args.cpkt)):
            unet.load_state_dict(
                load_file(os.path.join(args.repo, args.cpkt), device="cuda")
            )
        else:
            unet.load_state_dict(
                load_file(hf_hub_download(args.repo, args.cpkt), device="cuda")
            )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.base, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)

if pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast:
    pipe.upcast_vae()

# Compile the pipeline
if args.compiler == "oneflow":
    print("oneflow backend compile...")
    pipe = compile_pipe(
        pipe,
    )
    if args.load_graph:
        print("Loading graphs...")
        load_pipe(pipe, args.load_graph_dir)
elif args.compiler == "nexfort":
    print("nexfort backend compile...")
    nexfort_compiler_config = (
        json.loads(args.compiler_config) if args.compiler_config else None
    )

    options = nexfort_compiler_config
    pipe = compile_pipe(
        pipe, backend="nexfort", options=options, fuse_qkv_projections=True
    )
    if args.use_quantization and args.compiler == "nexfort":
        print("nexfort backend quant...")
        nexfort_quantize_config = (
            json.loads(args.quantize_config) if args.quantize_config else None
        )
        pipe = quantize_pipe(pipe, ignores=[], **nexfort_quantize_config)


with track_inference_time(warmup=True):
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=n_steps,
        guidance_scale=0,
        output_type=OUTPUT_TYPE,
    ).images


# Normal run
torch.manual_seed(args.seed)
with track_inference_time(warmup=False):
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=n_steps,
        guidance_scale=0,
        output_type=OUTPUT_TYPE,
    ).images


image[0].save(args.saved_image)

if args.save_graph:
    print("Saving graphs...")
    save_pipe(pipe, args.save_graph_dir)
