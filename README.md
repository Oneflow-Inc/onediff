[![PyPI version](https://badge.fury.io/py/onediff.svg)](https://badge.fury.io/py/onediff)
[![Docker image build](https://github.com/Oneflow-Inc/diffusers/actions/workflows/sd.yml/badge.svg)](https://github.com/Oneflow-Inc/diffusers/actions/workflows/sd.yml)

# OneFlow diffusers

OneFlow backend support for diffusers

## Performance

Updated on Nov 6, 2023.

|                | SD1.5 (512x512) | SD2.1 (512x512) | SDXL1.0-base（1024x1024） |
| -------------- | --------------- | --------------- | ------------------------- |
| 3090           | 42.38it/s       | 42.33it/s       | 6.66it/s                  |
| 4090           | 74.71it/s       | 73.57it/s       | 13.57it/s                 |
| A100-PCIE-40GB | 54.4it/s        | 54.06it/s       | 10.22it/s                 |
| A100-SXM4-80GB | 59.68it/s       | 61.91it/s       | 11.80it/s                 |

Run it by yourself using Docker:

```bash
docker pull oneflowinc/onediff:20231106
```


## Business inquiry

If you need ComfyUI or quant support or any other more advanced features, please send an email to caishenghang@oneflow.org to tell us about your use case and requirements!

|                      | OneFlow Open Source | OneFlow Pro |
| -------------------- | ------------------- | ----------- |
| diffusers API        | Yes                 | Yes         |
| UNet/VAE Compilation | Yes                 | Yes         |
| LoRA                 |                     | Yes         |
| ComfyUI              |                     | Yes         |
| Quantization         |                     | Yes         |
| Source Code Access   |                     | Yes         |
| Multiple Resolutions | Limited             | Yes         |
| Technical Support    | Community           | Yes         |

## Use within docker or install with pip

Please refer to this [wiki](https://github.com/Oneflow-Inc/diffusers/wiki/How-to-Run-OneFlow-Stable-Diffusion)

## Install from source

### Clone and install

```
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/master_open_source/cu118
python3 -m pip install "torch" "transformers==4.27.1" "diffusers[torch]==0.19.3"
python3 -m pip uninstall accelerate -y
git clone https://github.com/Oneflow-Inc/diffusers.git onediff
cd onediff && python3 -m pip install -e .
```

Other optional packages

CUDA 12.1

```bash
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/master_open_source/cu121
```

CUDA 12.2

```bash
python3 -m pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/master_open_source/cu122
```

### Login huggingface-cli

```
python3 -m pip install huggingface_hub
 ~/.local/bin/huggingface-cli login
```

## Run examples

```
python3 -m onediff.demo

or

python3 examples/text_to_image.py
```

## Release

- run examples to check it works

  ```bash
  python3 examples/text_to_image.py --model_id=...
  python3 examples/text_to_image_sdxl.py --base ...
  bash examples/unet_save_and_load.sh --model_id=...
  ```

- bump version in these files:

  ```
  setup.py
  src/onediff/__init__.py
  ```

- build wheel

  ```
  rm -rf dist
  python3 setup.py bdist_wheel
  ```

- upload to pypi

  ```bash
  twine upload dist/*
  ```

## More about OneFlow

OneFlow's main [repo](https://github.com/Oneflow-Inc/oneflow)
