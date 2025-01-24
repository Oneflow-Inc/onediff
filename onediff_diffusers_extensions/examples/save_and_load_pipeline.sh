#!/bin/bash

python3 examples/lightning/text_to_image_sdxl_light.py --base /share_nfs/hf_models/stable-diffusion-xl-base-1.0 --repo /share_nfs/hf_models/SDXL-Lightning --cpkt sdxl_lightning_4step_unet.safetensors --save_graph --save_graph_dir cached_unet_pipe

python3 examples/lightning/text_to_image_sdxl_light.py --base /share_nfs/hf_models/stable-diffusion-xl-base-1.0 --repo /share_nfs/hf_models/SDXL-Lightning --cpkt sdxl_lightning_4step_unet.safetensors --load_graph --load_graph_dir cached_unet_pipe


HF_HUB_OFFLINE=0 python3 examples/lightning/text_to_image_sdxl_light.py --base /share_nfs/hf_models/stable-diffusion-xl-base-1.0 --repo /share_nfs/hf_models/SDXL-Lightning --cpkt sdxl_lightning_4step_lora.safetensors  --save_graph --save_graph_dir cached_lora_pipe

HF_HUB_OFFLINE=0 python3 examples/lightning/text_to_image_sdxl_light.py --base /share_nfs/hf_models/stable-diffusion-xl-base-1.0 --repo /share_nfs/hf_models/SDXL-Lightning --cpkt sdxl_lightning_4step_lora.safetensors  --load_graph --load_graph_dir cached_lora_pipe
