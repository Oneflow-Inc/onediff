# Need `pip install -U peft transformers`

from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
import torch

pipe = DiffusionPipeline.from_pretrained(
    "/share_nfs/hf_models/sd-turbo",  # SD Turbo is a destilled version of Stable Diffusion 2.1
    #"stabilityai/sd-turbo",  # SD Turbo is a destilled version of Stable Diffusion 2.1
    # "stabilityai/stable-diffusion-2-1", # for the original stable diffusion 2.1 model
    torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")

from onediffx import compile_pipe

pipe = compile_pipe(pipe)

pipe.load_lora_weights("radames/sd-21-DPO-LoRA", adapter_name="dpo-lora-sd21")
pipe.set_adapters(["dpo-lora-sd21"], adapter_weights=[1.0]) # you can play with adapter_weights to increase the effect of the LoRA model
seed = 123123
prompt = "portrait headshot professional of elon musk"
negative_prompt = "3d render, cartoon, drawing, art, low light"
generator = torch.Generator().manual_seed(seed)

def run_once(dynamic_guidance_scale):
    for i in range(2):
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=512,
            height=512,
            num_inference_steps=2,
            generator=generator,
            guidance_scale=dynamic_guidance_scale,
            num_images_per_prompt=4
        ).images
        make_image_grid(images, 1, 4)

run_once(1.0)
# Adjust guidance_scale
# Will raise error
run_once(1.5)

