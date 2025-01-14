# Run IP-Adapter with OneDiff

1. [Environment Setup](#environment-setup)
   - [Set Up OneDiff](#set-up-onediff)
   - [Set Up OneFlow Backend](#set-up-oneflow-backend)
   - [Set Up NexFort Backend](#set-up-nexfort-backend)
   - [Set Up Diffusers Library](#set-up-diffusers)
   - [Set Up SDXL](#set-up-sdxl)
   - [Set Up IP-Adapter](#set-up-ip-adapter)
2. [Execution Instructions](#run)
   - [Run Without Compilation (Baseline)](#run-without-compilation-baseline)
   - [Run with oneflow backend compilation](#run-with-oneflow-backend-compilation)
   - [Run with nexfort backend compilation](#run-with-nexfort-backend-compilation)
3. [Performance Comparison](#performance-comparison)
4. [Dynamic Shape for IP-Adapter](#dynamic-shape-for-ip-adapter)

## Environment setup
### Set up onediff
https://github.com/siliconflow/onediff?tab=readme-ov-file#installation

### Set up oneflow backend
https://github.com/siliconflow/onediff?tab=readme-ov-file#oneflow

### Set up nexfort backend
https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort

### Set up diffusers

```
pip3 install --upgrade diffusers[torch]
```
### Set up SDXL
Model version for diffusers: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

HF pipeline: https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_xl.md

### Set up IP-Adapter
Models: https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models

Docs: https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter


## Run

### Run without compilation (Baseline)
```shell
python3 onediff_diffusers_extensions/examples/text_to_image_ip_adapter.py \
  --compiler none \
  --saved_image ip-adapter-none.png \
  --print-output \
  --multi-scale
```

### Run with oneflow backend compilation

```shell
python3 onediff_diffusers_extensions/examples/text_to_image_ip_adapter.py \
  --compiler oneflow \
  --saved_image ip-adapter-oneflow.png \
  --print-output \
  --multi-scale
```

### Run with nexfort backend compilation
```shell
python3 onediff_diffusers_extensions/examples/text_to_image_ip_adapter.py \
  --compiler nexfort \
  --saved_image ip-adapter-nexfort.png \
  --print-output \
  --multi-scale
```

## Performance comparison

Testing on NVIDIA GeForce RTX 3090 / 4090, with image size of 1024*1024, iterating 100 steps:
| Metric                                         | RTX 3090  1024*1024   | RTX 4090 1024*1024    |
| ---------------------------------------------- | --------------------- | --------------------- |
| Data update date (yyyy-mm-dd)                  | 2024-07-26            | 2024-07-26            |
| PyTorch iteration speed                        | 3.74 it/s             | 7.96 it/s             |
| OneDiff (oneflow) iteration speed              | 6.90 it/s (+84.5%)    | 14.45 it/s (+81.5%)   |
| OneDiff (nexfort) iteration speed              | 5.42 it/s (+44.9%)    | 11.97 it/s (+50.4%)   |
| PyTorch E2E time                               | 27.91 s               | 13.62 s               |
| OneDiff (oneflow) E2E time                     | 15.61 s (-44.1%)      | 7.88 s (-42.1%)       |
| OneDiff (nexfort) E2E time                     | 19.60 s (-29.8%)      | 9.28 s (-31.9%)       |
| PyTorch Max Mem Used                           | 14.58 GiB             | 14.59 GiB             |
| OneDiff (oneflow) Max Mem Used                 | 17.39 GiB             | 17.54 GiB             |
| OneDiff (nexfort) Max Mem Used                 | 15.10 GiB             | 15.09 GiB             |
| PyTorch Warmup with Run time                   |                       |                       |
| OneDiff (oneflow) Warmup with Compilation time | 131.20 s <sup>1</sup> | 76.07 s <sup>2</sup>  |
| OneDiff (nexfort) Warmup with Compilation time | 702.90 s <sup>1</sup> | 576.90 s <sup>2</sup> |
| OneDiff (oneflow) Warmup with Cache time       | N/A                   | N/A                   |
| OneDiff (nexfort) Warmup with Cache time       | 537.88 s <sup>1</sup> | 433.00 s <sup>2</sup> |

<sup>1</sup> OneDiff Warmup with Compilation time is tested on Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz. Note this is just for reference, and it varies a lot on different CPU.

<sup>2</sup> AMD EPYC 7543 32-Core Processor.


## Dynamic shape for IP-Adapter

Run:

```shell
python3 onediff_diffusers_extensions/examples/text_to_image_ip_adapter.py \
  --compiler oneflow \
  --saved_image ip-adapter-oneflow.png \
  --print-output \
  --multi-scale \
  --multi-resolution
```
## Note
When you set the scale for the ip-adapter, you should set it as a **`Tensor`** rather than a `float`. And if you want to modify scale, you should perform an in-place copy operation on the tensor associated with the scale, setting it to the new scale. For more details, please refer to the [code](../text_to_image_ip_adapter.py).

Here are the detailed reasons: If you set the scale as a float, during the compilation to generate the computation graph, the scale will be compiled as a constant, leading to the inability to modify the scale in the computation graph. If you set the scale as a `Tensor`, it will be compiled as a variable, and by modifying this tensor, you can modify the scale that is actually involved in the computations.
