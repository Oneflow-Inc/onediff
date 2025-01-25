### Default Configuration without LoRA

Run:
```
python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --saved-image sd.png
```

Performance:
- Iterations per second: 8.49 it/s
- Time taken: 3.92 seconds
- Max used CUDA memory: 10.465GiB



### Using LoRA

Run:
```
python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --saved-image sd_lora.png \
    --use_lora
```

Performance:
- Iterations per second: 8.53 it/s
- Time taken: 3.91 seconds
- Max used CUDA memory: 10.477GiB



### Compile

Run:
```
python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --compiler-config '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last", "options": {"triton.fuse_attention_allow_fp16_reduction": false}}' \
    --saved-image sd_lora_compile.png \
    --use_lora
```

Performance:
- Iterations per second: 14.94 it/s
- Time taken: 2.29 seconds
- Max used CUDA memory: 11.475GiB



### Compiled with Quantization

Run:
```
python3 onediff_diffusers_extensions/examples/quant_lora/test.py \
    --compiler-config '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last", "options": {"triton.fuse_attention_allow_fp16_reduction": false}}' \
    --quantize-config '{"quant_type": "int8_dynamic"}' \
    --saved-image sd_lora_int8.png \
    --use_lora
```

Performance:
- Iterations per second: 17.00 it/s
- Time taken: 2.04 seconds
- Max used CUDA memory: 8.808GiB
