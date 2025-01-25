# Reference: https://github.com/huggingface/diffusers/pull/9043
# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, revision="refs/pr/1"
# )
# python flux.py --height 1024 --width 1024 --base /data0/hf_models/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/93424e3a1530639fefdf08d2a7a954312e5cb254

import argparse
import time
import numpy as np
import torch
# Depends on the main branch of diffusers
from diffusers import FluxPipeline
from PIL import Image
parser = argparse.ArgumentParser()
# on A800-02
parser.add_argument("--base", type=str, default="black-forest-labs/FLUX.1-schnell")
parser.add_argument(
    "--prompt",
    type=str,
    default="anime scenery concept art, sunny day at beach, hyperrealistic!, fantasy art behance, huge dramatic brush strokes, heavens, by Takahashi Yuichi, view(full body + zoomed out), intricate and intense oil paint, panorama shot, incredible miyazaki, hyper realistic illustration, intricate dotart, header",
)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--n_steps", type=int, default=4)
parser.add_argument(
    "--saved_image", type=str, required=False, default="flux-out.png"
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--run", type=int, default=1)
parser.add_argument(
    "--compile", type=(lambda x: str(x).lower() in ["true", "1", "yes"]), default=True
)
args = parser.parse_args()
# load stable diffusion

# python flux.py --height 1024 --width 1024 --base /data0/hf_models/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/93424e3a1530639fefdf08d2a7a954312e5cb254
pipe = FluxPipeline.from_pretrained(args.base, torch_dtype=torch.bfloat16)

# pipe = FluxPipeline.from_pretrained(args.base, torch_dtype=torch.bfloat16, local_files_only=True)
# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
pipe.to("cuda")
if args.compile:
    import os
    os.environ['NEXFORT_FUSE_TIMESTEP_EMBEDDING'] = '0'
    #os.environ['NEXFORT_FX_FORCE_TRITON_SDPA'] = '1'
    #os.environ['NEXFORT_FX_FORCE_FA3_SDPA'] = '1'

    options = {"mode": "max-optimize:max-autotune:freezing:benchmark:low-precision"}
    #from onediffx import compile_pipe
    # pipe = compile_pipe(pipe, backend="nexfort", options=options)
    from onediff.infer_compiler import compile
    pipe.transformer = compile(pipe.transformer, backend="nexfort", options=options)
# generate image
generator = torch.manual_seed(args.seed)

#with torch.profiler.profile() as prof:
if True:
    print("Warmup")
    #with torch.profiler.record_function("flux warmup"):
    if True:
        for i in range(args.warmup):
            image = pipe(
                args.prompt,
                height=args.height,
                width=args.width,
                output_type="pil",
                num_inference_steps=args.n_steps, #use a larger number if you are using [dev]
                generator=torch.Generator("cpu").manual_seed(args.seed)
            ).images[0]
    print("Run")
    # with torch.profiler.record_function("flux compiled"):
    if True:
        for i in range(args.run):
            begin = time.time()
            image = pipe(
                args.prompt,
                height=args.height,
                width=args.width,
                output_type="pil",
                num_inference_steps=args.n_steps, #use a larger number if you are using [dev]
                generator=torch.Generator("cpu").manual_seed(args.seed)
            ).images[0]
            end = time.time()
            print(f"Inference time: {end - begin:.3f}s")
    image.save(f'{i=}th_{args.saved_image}.png')
# prof.export_chrome_trace("flux_compiled.json")

# print("New size")
# image = pipe(
#     args.prompt,
#     height=args.height // 2,
#     width=args.width // 2,
#     output_type="pil",
#     num_inference_steps=args.n_steps, #use a larger number if you are using [dev]
#     generator=torch.Generator("cpu").manual_seed(args.seed)
# ).images[0]