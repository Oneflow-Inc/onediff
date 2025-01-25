_IS_NPU_AVAILABLE = False
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    _IS_NPU_AVAILABLE = True
except:
    pass


if _IS_NPU_AVAILABLE:
    import comfy
    from comfy.model_management import (
        is_device_cpu,
        is_intel_xpu,
        ENABLE_PYTORCH_ATTENTION,
    )

    torch_npu.npu.set_compile_mode(jit_compile=False)

    def patch_pytorch_attention_flash_attention():
        if ENABLE_PYTORCH_ATTENTION:
            return True
        return False

    def patch_get_free_memory(dev=None, torch_free_too=False):
        # stats = torch.npu.memory_stats(dev)
        # mem_active = stats['active_bytes.all.current']
        # mem_reserved = stats['reserved_bytes.all.current']
        # mem_free_npu, _ = torch.npu.mem_get_info(dev)
        # mem_free_torch = mem_reserved - mem_active
        # mem_free_total = mem_free_npu + mem_free_torch
        mem_free_total = 48 * 1024 * 1024 * 1024  # TODO
        mem_free_torch = mem_free_total

        if torch_free_too:
            return (mem_free_total, mem_free_torch)
        else:
            return mem_free_total

    def patch_should_use_fp16(
        device=None, model_params=0, prioritize_performance=True, manual_cast=False
    ):
        if device is not None:
            if is_device_cpu(device):
                return False
        return True

    def patch_should_use_bf16(
        device=None, model_params=0, prioritize_performance=True, manual_cast=False
    ):
        return False

    comfy.model_management.pytorch_attention_flash_attention = (
        patch_pytorch_attention_flash_attention
    )
    comfy.model_management.get_free_memory = patch_get_free_memory
    comfy.model_management.should_use_fp16 = patch_should_use_fp16
    comfy.model_management.should_use_bf16 = patch_should_use_bf16
