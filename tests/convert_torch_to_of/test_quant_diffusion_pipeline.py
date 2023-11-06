import pytest
import torch
from onediff.infer_compiler.convert_torch_to_of.mock_diffusers_quant import (
    QuantDiffusionPipeline,
)
from onediff.infer_compiler import oneflow_compile


@pytest.mark.parametrize("model", ["/ssd/home/hanbinbin/sdxl-1.0-base-int8"])
@pytest.mark.parametrize("fake_quant", [False])
@pytest.mark.parametrize("static", [False])
@pytest.mark.parametrize("bits", [8])
@pytest.mark.parametrize("graph", [True])
@pytest.mark.parametrize(
    "prompt",
    ['"street style, detailed, raw photo, woman, face, shot on CineStill 800T"'],
)
def test_quant_diffusion_pipeline(model, fake_quant, static, bits, graph, prompt):
    pipe = QuantDiffusionPipeline.from_pretrained(
        model, fake_quant, static, bits, graph
    )
    pipe.to("cuda")
    pipe.unet = oneflow_compile(pipe.unet)
    pipe(prompt, height=512, width=512)
