import argparse
import json
import time
from typing import List, Union

import imageio
import numpy as np

import PIL
import torch

from diffusers import CogVideoXPipeline
from onediffx import compile_pipe, quantize_pipe


def export_to_video_imageio(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 8,
) -> str:
    """
    Export the video frames to a video file using imageio lib to Avoid "green screen" issue (for example CogVideoX)
    """
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    if isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)
    return output_video_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use onediif to accelerate image generation with CogVideoX"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="Model path or identifier.",
    )
    parser.add_argument(
        "--compiler",
        type=str,
        default="none",
        help="Compiler backend to use. Options: 'none', 'nexfort', 'torch'",
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
        default="In the haunting backdrop of a war-torn city, where ruins and crumbled walls tell a story of devastation, a poignant close-up frames a young girl. Her face is smudged with ash, a silent testament to the chaos around her. Her eyes glistening with a mix of sorrow and resilience, capturing the raw emotion of a world that has lost its innocence to the ravages of conflict.",
        help="Prompt for the image generation.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.5,
        help="The scale factor for the guidance.",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=50, help="Number of inference steps."
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate per prompt",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output.mp4",
        help="The path where the generated video will be saved",
    )
    parser.add_argument(
        "--seed", type=int, default=66, help="Seed for random number generation."
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Number of warm-up iterations before actual inference.",
    )
    return parser.parse_args()


args = parse_args()

device = torch.device("cuda")


class CogVideoGenerator:
    def __init__(
        self, model, compiler_config=None, quantize_config=None, compiler="none"
    ):
        self.pipe = CogVideoXPipeline.from_pretrained(
            model, torch_dtype=torch.float16
        ).to(device)

        self.pipe.enable_model_cpu_offload()

        self.prompt_embeds = None

        if compiler == "nexfort":
            if compiler_config:
                print("nexfort backend compile...")
                self.pipe = self.compile_pipe(self.pipe, compiler_config)

            if quantize_config:
                print("nexfort backend quant...")
                self.pipe = self.quantize_pipe(self.pipe, quantize_config)

        elif compiler == "torch":
            self.pipe.transformer = torch.compile(
                self.pipe.transformer, mode="max-autotune", fullgraph=True
            )
            # self.pipe.vae.decode = torch.compile(
            #     self.pipe.vae.decode, mode="max-autotune", fullgraph=True
            # )

    def encode_prompt(self, prompt, num_videos_per_prompt):
        self.prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=226,
            device=device,
            dtype=torch.float16,
        )

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
        video = self.pipe(**gen_args).frames[0]
        end_time = time.time()

        export_to_video_imageio(video, args.output_path, fps=8)

        return video, end_time - start_time

    def compile_pipe(self, pipe, compiler_config):
        options = compiler_config
        pipe = compile_pipe(
            pipe,
            backend="nexfort",
            options=options,
            ignores=["vae"],
            fuse_qkv_projections=True,
        )
        return pipe

    def quantize_pipe(self, pipe, quantize_config):
        pipe = quantize_pipe(pipe, ignores=[], **quantize_config)
        return pipe


def main():
    nexfort_compiler_config = (
        json.loads(args.compiler_config) if args.compiler_config else None
    )
    nexfort_quantize_config = (
        json.loads(args.quantize_config) if args.quantize_config else None
    )

    CogVideo = CogVideoGenerator(
        args.model,
        nexfort_compiler_config,
        nexfort_quantize_config,
        compiler=args.compiler,
    )

    CogVideo.encode_prompt(args.prompt, args.num_videos_per_prompt)

    gen_args = {
        # "prompt": args.prompt,
        "prompt_embeds": CogVideo.prompt_embeds,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "negative_prompt_embeds": torch.zeros_like(
            CogVideo.prompt_embeds
        ),  # Not Supported negative prompt
        # "num_frames": 8,
    }

    CogVideo.warmup(gen_args, args.warmup_iterations)
    torch.cuda.empty_cache()

    _, inference_time = CogVideo.generate(gen_args)
    torch.cuda.empty_cache()
    print(
        f"Generated video saved to {args.output_path} in {inference_time:.2f} seconds."
    )
    cuda_mem_after_used = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Max used CUDA memory : {cuda_mem_after_used:.3f}GiB")


if __name__ == "__main__":
    main()
