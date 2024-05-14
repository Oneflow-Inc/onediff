## Accelerating ComfyUI_IPAdapter_plus with OneDiff
### Environment
Please Refer to the Readme in the Respective Repositories for Installation Instructions.

- ComfyUI:
  - github: https://github.com/comfyanonymous/ComfyUI
  - commit: 4bd7d55b9028d79829a645edfe8259f7b7a049c0 
  - Date: Thu Apr 11 22:43:05 2024 -0400
  
- ComfyUI_IPAdapter_plus:
  - github: https://github.com/cubiq/ComfyUI_IPAdapter_plus
  - commit 417d806e7a2153c98613e86407c1941b2b348e88 
  - Date:  Wed Apr 10 13:28:41 2024 +0200

- ComfyUI_InstantID$ git log
  - commit e9cc7597b2a7cd441065418a975a2de4aa2450df 
  - Date:   Tue Apr 9 14:05:15 2024 +0200

- ComfyUI-AnimateDiff-Evolved$ git log
  - commit f9e0343f4c4606ee6365a9af4a7e16118f1c45e1 
  - Date:   Sat Apr 6 17:32:15 2024 -0500

- OneDiff:
  - github: https://github.com/siliconflow/onediff 

### Quick Start

> Recommend running the official example of ComfyUI_IPAdapter_plus now, and then trying OneDiff acceleration. 

Experiment (NVIDIA A100-PCIE-40GB) Workflow for OneDiff Acceleration in ComfyUI_IPAdapter_plus:

1. Replace the **`Load Checkpoint`** node with **`Load Checkpoint - OneDiff`** node. 
2. Add a **`Batch Size Patcher`** node before the **`Ksampler`** node (due to temporary lack of support for dynamic batch size).
As follows:

![ipadapter_example](https://github.com/siliconflow/onediff/assets/109639975/adb2df92-b0f0-4650-ae02-5fd458209b92)

![ipadapter_example int8](https://github.com/siliconflow/onediff/assets/109639975/009b2f84-37f2-4b29-a2be-d1091da00b98)

 | PyTorch                                                                                                | OneDiff                                                                                                  |
 | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
 | ![torch](https://github.com/siliconflow/onediff/assets/109639975/b99838a6-2809-4e70-a4f2-966ba76c69d6) | ![onediff](https://github.com/siliconflow/onediff/assets/109639975/455741aa-d4e7-4b43-bfac-c5c52a66ac12) |

- NVIDIA A100-PCIE-40GB 
- batch_size 4
- warmup 4
- e2e
  - torch: 2.08 s (baseline)
  - onediff: 1.36 s (percentage improvement ~34.6%)
  - onediff int8: 1.20 s (percentage improvement ~42.3%) 


### Compatibility

| Functionality      | Supported |
| ------------------ | --------- |
| Dynamic Shape      | Yes       |
| Dynamic Batch Size | No        |
| Vae Speed Up       | Yes       |

## Contact

For users of OneDiff Community, please visit [GitHub Issues](https://github.com/siliconflow/onediff/issues) for bug reports and feature requests.

For users of OneDiff Enterprise, you can contact contact@siliconflow.com for commercial support.

Feel free to join our [Discord](https://discord.gg/RKJTjZMcPQ) community for discussions and to receive the latest updates.