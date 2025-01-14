import argparse
import inspect
import json
import os
import time
from pathlib import Path

import torch

from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
from onediffx import compile_pipe, load_pipe, save_pipe

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
    "--base", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--ipadapter", type=str, default="h94/IP-Adapter")
parser.add_argument("--subfolder", type=str, default="sdxl_models")
parser.add_argument("--weight_name", type=str, default="ip-adapter_sdxl.bin")
parser.add_argument(
    "--input_image",
    type=str,
    default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png",
)
parser.add_argument(
    "--prompt",
    default="a polar bear sitting in a chair drinking a milkshake",
    help="Prompt",
)
parser.add_argument(
    "--negative-prompt",
    default="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
    help="Negative prompt",
)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--n_steps", type=int, default=100)
parser.add_argument(
    "--saved_image", type=str, required=False, default="ip-adapter-out.png"
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument(
    "--compiler", type=str, default="oneflow", choices=["none", "nexfort", "oneflow"]
)
parser.add_argument("--compile-options", type=str, default=nexfort_options)
parser.add_argument("--cache-dir", default="./onediff_cache", help="cache directory")
parser.add_argument("--multi-scale", action="store_true")
parser.add_argument("--multi-resolution", action="store_true")
parser.add_argument("--print-output", action="store_true")
args = parser.parse_args()


class IterationProfiler:
    def __init__(self):
        self.begin = None
        self.end = None
        self.num_iterations = 0

    def get_iter_per_sec(self):
        if self.begin is None or self.end is None:
            return None
        self.end.synchronize()
        dur = self.begin.elapsed_time(self.end)
        return self.num_iterations / dur * 1000.0

    def callback_on_step_end(self, pipe, i, t, callback_kwargs={}):
        if self.begin is None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.begin = event
        else:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.end = event
            self.num_iterations += 1
        return callback_kwargs


# load an image
ip_adapter_image = load_image(args.input_image)

# load stable diffusion and ip-adapter
pipe = AutoPipelineForText2Image.from_pretrained(
    args.base,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.load_ip_adapter(
    args.ipadapter, subfolder=args.subfolder, weight_name=args.weight_name
)

# Set ipadapter scale as a tensor instead of a float
# If scale is a float, it cannot be modified after the graph is traced
ipadapter_scale = torch.tensor(args.scale, dtype=torch.float, device="cuda")
pipe.set_ip_adapter_scale(ipadapter_scale)
pipe.to("cuda")


cache_path = os.path.join(args.cache_dir, type(pipe).__name__)

if args.compiler == "none":
    pass
elif args.compiler == "nexfort":
    compile_options = args.compile_options
    if isinstance(compile_options, str):
        compile_options = json.loads(compile_options)
    if args.multi_resolution:
        compile_options["dynamic"] = True
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "./.torchinductor")
    pipe = compile_pipe(pipe, backend="nexfort", options=compile_options)
else:
    pipe = compile_pipe(pipe, backend="oneflow")
    if os.path.exists(cache_path):
        # TODO(WangYi): load pipe has bug here, which makes scale unchangeable
        # load_pipe(pipe, cache_path)
        pass


# NOTE: Warm it up.
# The initial calls will trigger compilation and might be very slow.
# After that, it should be very fast.
if args.warmup > 0:
    begin = time.time()
    print("=======================================")
    print("Begin warmup")
    for _ in range(args.warmup):
        pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            ip_adapter_image=ip_adapter_image,
            num_inference_steps=args.n_steps,
        )
    end = time.time()
    print("End warmup")
    print(f"Warmup time: {end - begin:.3f}s")
    print("=======================================")

# Let"s see it!
# Note: Progress bar might work incorrectly due to the async nature of CUDA.
kwarg_inputs = dict(
    prompt=args.prompt,
    ip_adapter_image=ip_adapter_image,
    negative_prompt=args.negative_prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.n_steps,
    generator=torch.Generator(device="cpu").manual_seed(0),
)
iter_profiler = IterationProfiler()
if "callback_on_step_end" in inspect.signature(pipe).parameters:
    kwarg_inputs["callback_on_step_end"] = iter_profiler.callback_on_step_end
elif "callback" in inspect.signature(pipe).parameters:
    kwarg_inputs["callback"] = iter_profiler.callback_on_step_end
begin = time.time()
image_to_print = pipe(**kwarg_inputs).images[0]
image_to_print.save("result.png")
image_path = (
    f"{Path(args.saved_image).stem}_{args.scale}_{args.compiler}"
    + Path(args.saved_image).suffix
)
print(f"save output image to {image_path}")
image_to_print.save(image_path)
end = time.time()

print("=======================================")
print(f"Inference time: {end - begin:.3f}s")
iter_per_sec = iter_profiler.get_iter_per_sec()
if iter_per_sec is not None:
    print(f"Iterations per second: {iter_per_sec:.3f}")
if args.compiler == "oneflow":
    import oneflow as flow  # usort: skip

    cuda_mem_after_used = flow._oneflow_internal.GetCUDAMemoryUsed() / 1024
else:
    cuda_mem_after_used = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Max used CUDA memory : {cuda_mem_after_used:.3f}GiB")
print("=======================================")


if args.multi_scale:
    scales = [0.1, 0.5, 1]
    for scale in scales:
        # Use ipadapter_scale.copy_ instead of pipeline.set_ip_adapter_scale to modify scale
        ipadapter_scale.copy_(torch.tensor(scale, dtype=torch.float, device="cuda"))
        pipe.set_ip_adapter_scale(ipadapter_scale)
        image = pipe(**kwarg_inputs).images[0]
        image_path = (
            f"{Path(args.saved_image).stem}_{scale}" + Path(args.saved_image).suffix
        )
        print(f"save output image to {image_path}")
        image.save(image_path)

if args.multi_resolution:
    from itertools import product

    sizes = [1024, 512, 768, 256]
    for h, w in product(sizes, sizes):
        image = pipe(
            prompt=args.prompt,
            ip_adapter_image=ip_adapter_image,
            negative_prompt=args.negative_prompt,
            height=h,
            width=w,
            num_inference_steps=args.n_steps,
            generator=torch.Generator(device="cpu").manual_seed(0),
        ).images[0]
        print(f"Running at resolution: {h}x{w}")

if args.print_output:
    from onediff.utils.import_utils import is_nexfort_available

    if is_nexfort_available():
        from nexfort.utils.term_image import print_image

        print_image(image_to_print, max_width=80)


if args.compiler == "oneflow":
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    save_pipe(pipe, cache_path)
