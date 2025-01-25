import json
import os
from typing import NamedTuple

from core.registry import create_generator_registry
from core.service_client import ComfyGraph

WORKFLOW_DIR = "resources/workflows"
FACE_IMAGE_DIR = "/share_nfs/hf_models/comfyui_resources/input/faces"
POSE_IMAGE_DIR = "/share_nfs/hf_models/comfyui_resources/input/poses"
SDXL_MODELS = [
    "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    "Pony_Realism.safetensors",
    "sdxl/dreamshaperXL_v21TurboDPMSDE.safetensors",
]
SD1_5_MODELS = [
    "sd15/020.realisticVisionV51_v51VAE.safetensors",
    "sd15/majicmixRealistic_v7.safetensors",
    "sd15/v1-5-pruned-emaonly.ckpt",
    "sd15/helloyoung25d_V10f.safetensors",
    "sd15/RealCartoonSpecialPruned.safetensors",
]


class InputParams(NamedTuple):
    graph: ComfyGraph


def read_prompts(file_path="resources/prompts.txt"):
    with open(file_path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
    return [line.strip() for line in lines if line.strip()]


def get_all_images(
    directory=FACE_IMAGE_DIR,
    image_extensions=set([".jpg", ".jpeg", ".png", ".gif", ".bmp"]),
):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory)]
    image_files = [
        f
        for f in all_files
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in image_extensions
    ]
    return image_files


# Create register and get functions
register_generator, dispatch_generator = create_generator_registry()


@register_generator(f"{WORKFLOW_DIR}/example_workflow_api.json")
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)
    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])
    for height in [1024, 768, 512]:
        for width in [1024, 768, 512]:
            graph.set_image_size(height=height, width=width)
            yield InputParams(graph=graph)


@register_generator(
    [f"{WORKFLOW_DIR}/baseline/txt2img.json", f"{WORKFLOW_DIR}/oneflow/txt2img.json"]
)
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)
    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])
    for sdxl_model in SDXL_MODELS:
        graph.set_image_size(height=1024, width=1024)
        graph.graph["31"]["inputs"]["ckpt_name"] = sdxl_model
        yield InputParams(graph)
    for sd1_5_model in SD1_5_MODELS:
        graph.set_image_size(height=768, width=512)
        graph.graph["31"]["inputs"]["ckpt_name"] = sd1_5_model
        yield InputParams(graph)


SD3_WORKFLOWS = [
    f"{WORKFLOW_DIR}/baseline/sd3_basic.json",
    f"{WORKFLOW_DIR}/nexfort/sd3_basic.json",
]


@register_generator(SD3_WORKFLOWS)
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)

    graph = ComfyGraph(graph=workflow, sampler_nodes=["271"])
    texts = read_prompts()
    for height in [1024, 768, 512]:
        for width in [1024, 768, 512]:
            for text in texts[-5:]:
                graph.set_prompt(prompt=text)
                graph.set_image_size(height=height, width=width)
                yield InputParams(graph=graph)


@register_generator(f"{WORKFLOW_DIR}/oneflow/sdxl-control-lora-speedup.json")
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)

    graph = ComfyGraph(graph=workflow, sampler_nodes=["1"])
    yield InputParams(graph=graph)


@register_generator(
    [
        f"{WORKFLOW_DIR}/baseline/ComfyUI_IPAdapter_plus/ipadapter_advanced.json",
        f"{WORKFLOW_DIR}/oneflow/ComfyUI_IPAdapter_plus/ipadapter_advanced.json",
    ]
)
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)

    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])
    for image in get_all_images():
        graph.graph["12"]["inputs"]["image"] = image
        for height in [768, 512]:
            for width in [768, 512]:
                positive_prompt = "in a peaceful spring morning a woman wearing a white shirt is sitting in a park on a bench\n\nhigh quality, detailed, diffuse light"
                negative_prompt = "blurry, noisy, messy, lowres, jpeg, artifacts, ill, distorted, malformed"
                graph.set_prompt(positive_prompt, negative_prompt)
                graph.set_image_size(height=height, width=width)
                yield InputParams(graph=graph)


@register_generator(
    [
        f"{WORKFLOW_DIR}/baseline/lora.json",
        f"{WORKFLOW_DIR}/baseline/lora_multiple.json",
        f"{WORKFLOW_DIR}/oneflow/lora_speedup.json",
        f"{WORKFLOW_DIR}/oneflow/lora_multiple_speedup.json",
    ]
)
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)

    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])
    graph.set_prompt("masterpiece best quality girl, hanfu", "bad hands")
    root_path = "sd15/"
    checkpoint_nodes = []
    for node in graph.graph.values():
        if node["class_type"] in [
            "CheckpointLoaderSimple",
            "OneDiffCheckpointLoaderSimple",
        ]:
            checkpoint_nodes.append(node)
    assert len(checkpoint_nodes) == 1

    for file_name in [
        "020.realisticVisionV51_v51VAE.safetensors",
        "v1-5-pruned-emaonly.ckpt",
    ]:
        checkpoint_path = os.path.join(root_path, file_name)
        checkpoint_nodes[0]["inputs"]["ckpt_name"] = checkpoint_path
        yield InputParams(graph=graph)


@register_generator(
    [
        f"{WORKFLOW_DIR}/baseline/ComfyUI_InstantID/instantid_posed.json",
        f"{WORKFLOW_DIR}/oneflow/ComfyUI_InstantID/instantid_posed_speedup.json",
    ]
)
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)
    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])

    face_imgs = get_all_images(FACE_IMAGE_DIR)[:2]
    pose_imgs = get_all_images(POSE_IMAGE_DIR)[:2]
    for face_img in face_imgs:
        for pose_img in pose_imgs:
            # print(f'{face_img=} {pose_img=}')
            graph.graph["13"]["inputs"]["image"] = face_img
            graph.graph["67"]["inputs"]["image"] = pose_img
            yield InputParams(graph=graph)


@register_generator(
    [
        f"{WORKFLOW_DIR}/baseline/PuLID_ComfyUI/PuLID_4-Step_lightning.json",
        f"{WORKFLOW_DIR}/oneflow/PuLID_ComfyUI/PuLID_4-Step_lightning.json",
    ]
)
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)
    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])

    for height in [768, 512]:
        for width in [768]:
            for face_img in get_all_images(FACE_IMAGE_DIR):
                graph.graph["12"]["inputs"]["image"] = face_img
                graph.set_image_size(height=height, width=width)
                yield InputParams(graph=graph)


@register_generator(
    [
        f"{WORKFLOW_DIR}/baseline/PuLID_ComfyUI/PuLID_IPAdapter_style_transfer.json",
        f"{WORKFLOW_DIR}/oneflow/PuLID_ComfyUI/PuLID_IPAdapter_style_transfer.json",
    ]
)
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)
    graph = ComfyGraph(graph=workflow, sampler_nodes=["3"])
    face_imgs = get_all_images(FACE_IMAGE_DIR)[:2]
    pose_imgs = get_all_images(POSE_IMAGE_DIR)[:2]
    for face_img in face_imgs:
        for pose_img in pose_imgs:
            # print(f'{face_img=} {pose_img=}')
            graph.graph["12"]["inputs"]["image"] = face_img
            graph.graph["48"]["inputs"]["image"] = pose_img
            yield InputParams(graph=graph)


@register_generator(
    [
        f"{WORKFLOW_DIR}/baseline/flux1-schnell.json",
        f"{WORKFLOW_DIR}/nexfort/flux1-schnell.json",
    ]
)
def _(workflow_path, *args, **kwargs):
    with open(workflow_path, "r") as fp:
        workflow = json.load(fp)
    graph = ComfyGraph(graph=workflow, sampler_nodes=["13"])
    texts = read_prompts()
    resolutions_and_ratios = [
        (1024, 1024, "1:1 Square"),
        (1152, 896, "9:7"),
        (896, 1152, "7:9"),
        (1216, 832, "19:13"),
        (832, 1216, "13:19"),
        (1344, 768, "7:4 Horizontal"),
        (768, 1344, "4:7 Vertical"),
        (1536, 640, "12:5 Horizontal"),
        (640, 1536, "5:12 Vertical, the closest to the iPhone resolution")
    ]
    for height, width, _ in resolutions_and_ratios:
        for batch_size in [1,2]:
            for text in texts[-5:]:
                graph.graph["6"]["inputs"]["text"] = text
                graph.set_image_size(height=height, width=width, batch_size=batch_size)
                yield InputParams(graph=graph)