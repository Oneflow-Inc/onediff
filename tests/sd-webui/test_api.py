import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from utils import (
    cal_ssim,
    check_and_generate_images,
    dump_image,
    get_all_args,
    get_base_args,
    get_data_summary,
    get_image_array_from_response,
    get_target_image_filename,
    get_threshold,
    IMG2IMG_API_ENDPOINT,
    is_txt2img,
    OPTIONS_API_ENDPOINT,
    post_request_and_check,
    SAVED_GRAPH_NAME,
    TXT2IMG_API_ENDPOINT,
    WEBUI_SERVER_URL,
)


@pytest.fixture(scope="session", autouse=True)
def change_model():
    option_payload = {
        "sd_model_checkpoint": "AWPainting_v1.2.safetensors",
    }
    post_request_and_check(f"{WEBUI_SERVER_URL}/{OPTIONS_API_ENDPOINT}", option_payload)


@pytest.fixture(scope="session", autouse=True)
def prepare_target_images():
    print("checking if target images exist...")
    check_and_generate_images()


@pytest.fixture()
def base_url():
    return WEBUI_SERVER_URL


@pytest.fixture()
def url_txt2img(base_url):
    return f"{base_url}/{TXT2IMG_API_ENDPOINT}"


@pytest.fixture()
def url_img2img(base_url):
    return f"{base_url}/{IMG2IMG_API_ENDPOINT}"


@pytest.fixture()
def url_set_config(base_url):
    return f"{base_url}/{OPTIONS_API_ENDPOINT}"


@pytest.mark.parametrize("data", get_all_args())
def test_image_ssim(base_url, data):
    print(f"testing: {get_data_summary(data)}")
    endpoint = TXT2IMG_API_ENDPOINT if is_txt2img(data) else IMG2IMG_API_ENDPOINT
    url = f"{base_url}/{endpoint}"
    generated_image = get_image_array_from_response(post_request_and_check(url, data))
    target_image_path = get_target_image_filename(data)
    target_image = np.array(Image.open(target_image_path))
    ssim_value = cal_ssim(generated_image, target_image)
    if ssim_value < get_threshold(data):
        dump_image(target_image, generated_image, Path(target_image_path).name)
    assert ssim_value > get_threshold(data)


def test_onediff_save_graph(url_txt2img):
    script_args = {
        "script_args": [
            False,  # quantization
            None,  # graph_checkpoint
            SAVED_GRAPH_NAME,  # saved_graph_name
        ]
    }
    data = {**get_base_args(), **script_args}
    post_request_and_check(url_txt2img, data)


def test_onediff_load_graph(url_txt2img):
    script_args = {
        "script_args": [
            False,  # quantization
            SAVED_GRAPH_NAME,  # graph_checkpoint
            "",  # saved_graph_name
        ]
    }
    data = {**get_base_args(), **script_args}
    post_request_and_check(url_txt2img, data)


@pytest.mark.skip
def test_onediff_refiner(url_txt2img):
    extra_args = {
        "sd_model_checkpoint": "sd_xl_base_1.0.safetensors",
        "refiner_checkpoint": "sd_xl_refiner_1.0.safetensors [7440042bbd]",
        "refiner_switch_at": 0.8,
    }
    data = {**get_base_args(), **extra_args}
    # loop 3 times for checking model switching between base and refiner
    for _ in range(3):
        post_request_and_check(url_txt2img, data)
