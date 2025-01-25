import argparse
import json
import time

import torch

from diffusers import AutoPipelineForText2Image as pipeline_cls
from onediffx import compile_pipe, quantize_pipe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use onediif (nexfort) to accelerate image generation with Stable Diffusion + LoRA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model path or identifier.",
    )
    parser.add_argument(
        "--lora-model-id",
        type=str,
        default="minimaxir/sdxl-wrong-lora",
        help="LoRA model identifier for fine-tuning weights.",
    )
    parser.add_argument(
        "--lora-filename",
        type=str,
        default="pytorch_lora_weights.bin",
        help="Filename for LoRA weights.",
    )
    parser.add_argument(
        "--compiler-config", type=str, help="JSON string for compiler config."
    )
    parser.add_argument(
        "--quantize-config", type=str, help="JSON string for quantization config."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="anime style, 1 girl, indoors, sitting on the sofa, living room, pink hair, white sock, blue eyes, from back, from above, face towards viewer, playing video games, holding controller, black silk, parted lips.",
        help="Prompt for the image generation.",
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Height of the generated image."
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Width of the generated image."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="The scale factor for the guidance.",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=30, help="Number of inference steps."
    )
    parser.add_argument(
        "--saved-image",
        type=str,
        default="./sd.png",
        help="Path to save the generated image.",
    )
    parser.add_argument(
        "--seed", type=int, default=888, help="Seed for random number generation."
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Number of warm-up iterations before actual inference.",
    )
    parser.add_argument(
        "--use_lora", action="store_true", help="Use LoRA weights for the generation"
    )
    return parser.parse_args()


args = parse_args()

device = torch.device("cuda")


class SDGenerator:
    def __init__(self, model, compiler_config=None, quantize_config=None):
        self.pipe = pipeline_cls.from_pretrained(model, torch_dtype=torch.float16)

        if args.use_lora:
            print("Use LoRA...")
            self.pipe.load_lora_weights(
                args.lora_model_id, weight_name=args.lora_filename
            )
            self.pipe.fuse_lora()

        self.pipe.to(device)

        if compiler_config:
            print("compile...")
            self.pipe = self.compile_pipe(self.pipe, compiler_config)

        if quantize_config:
            print("quant...")
            self.pipe = self.quantize_pipe(self.pipe, quantize_config)

    def warmup(self, gen_args, warmup_iterations):
        warmup_args = gen_args.copy()

        warmup_args["generator"] = torch.Generator(device=device).manual_seed(0)

        print("Starting warmup...")
        start_time = time.time()
        for _ in range(warmup_iterations):
            self.pipe(**warmup_args)
        end_time = time.time()
        print("Warmup complete.")
        print(f"Warmup time: {end_time - start_time:.2f} seconds")

    def generate(self, gen_args):
        gen_args["generator"] = torch.Generator(device=device).manual_seed(args.seed)

        # Run the model
        start_time = time.time()
        images = self.pipe(**gen_args).images
        end_time = time.time()

        images[0].save(args.saved_image)

        return images[0], end_time - start_time

    def compile_pipe(self, pipe, compiler_config):
        options = compiler_config
        pipe = compile_pipe(
            pipe, backend="nexfort", options=options, fuse_qkv_projections=True
        )
        return pipe

    def quantize_pipe(self, pipe, quantize_config):
        pipe = quantize_pipe(pipe, ignores=[], **quantize_config)
        return pipe


def main():
    compiler_config = json.loads(args.compiler_config) if args.compiler_config else None
    quantize_config = json.loads(args.quantize_config) if args.quantize_config else None

    sd = SDGenerator(args.model, compiler_config, quantize_config)

    gen_args = {
        "prompt": args.prompt,
        "negative_prompt": "wrong",
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
    }

    sd.warmup(gen_args, args.warmup_iterations)

    image, inference_time = sd.generate(gen_args)
    print(
        f"Generated image saved to {args.saved_image} in {inference_time:.2f} seconds."
    )
    cuda_mem_after_used = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Max used CUDA memory : {cuda_mem_after_used:.3f}GiB")


if __name__ == "__main__":
    main()
