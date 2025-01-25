import argparse
import json
import os
import sys
import time
from io import BytesIO
from typing import List, NamedTuple, Union

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

sys.path.append("./src")

from core.log_utils import setup_logging
from core.service_client import comfy_client_context
from input_registration import dispatch_generator

DEFAULT_HOST = "127.0.0.1"
DEFAULT_COMFY_PORT = "8188"


def parse_args():
    parser = argparse.ArgumentParser(description="Test ComfyUI workflow")
    parser.add_argument(
        "-w", "--workflow", type=str, nargs="+", required=True, help="Workflow file(s)"
    )
    parser.add_argument(
        "--listen", type=str, default=DEFAULT_HOST, help="Service listen address"
    )
    parser.add_argument(
        "--comfy-port",
        type=str,
        default=DEFAULT_COMFY_PORT,
        help="ComfyUI service port",
    )
    parser.add_argument(
        "--output-images", action="store_true", help="Enable output of images."
    )
    parser.add_argument(
        "--baseline-dir", type=str, help="Directory for baseline output."
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.5,
        help="SSIM threshold for image comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output results.",
    )
    parser.add_argument(
        "--exp-name", type=str, default="exp", help="Experiment name for logging."
    )
    return parser.parse_args()


def save_image(image_data: bytes, output_dir: str, image_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, image_name)
    Image.open(BytesIO(image_data)).save(image_path)
    return image_path


def calculate_ssim(image1: Image.Image, image2: Image.Image) -> float:
    image1_np = np.array(image1.convert("RGB"))
    image2_np = np.array(image2.convert("RGB"))
    assert image1_np.shape == image2_np.shape, "Images must have the same dimensions"
    return ssim(image1_np, image2_np, channel_axis=2)


class ImageInfo(NamedTuple):
    height: int
    width: int
    image_path: str = ""


class WorkflowProcessor:
    def __init__(
        self,
        output_images: bool,
        output_dir: str,
        logger,
    ):
        self.output_images = output_images
        self.output_dir = output_dir
        self.logger = logger

    def process_image(self, image_data: bytes, index: int) -> ImageInfo:
        pil_image = Image.open(BytesIO(image_data))
        self.logger.info(
            f"Image Size - Height: {pil_image.height}px, Width: {pil_image.width}px"
        )
        assert np.array(pil_image).any() != 0, "Image is blank"

        image_path = ""
        if self.output_images:
            image_path = save_image(image_data, self.output_dir, f"image_{index}.png")
            self.logger.info(f"Saved image to: {image_path}")

        return ImageInfo(
            height=pil_image.height, width=pil_image.width, image_path=image_path
        )


def run_workflow(
    workflow: Union[str, List[str]],
    comfy_port: str = DEFAULT_COMFY_PORT,
    output_images: bool = True,
    baseline_dir: str = None,
    ssim_threshold: float = 0.5,
    output_dir: str = "results",
    exp_name: str = "exp",
) -> None:
    logger, result_dir = setup_logging(output_dir, exp_name=exp_name)
    logger.info(f"Result directory: {result_dir}")

    processor = WorkflowProcessor(
        output_images,
        os.path.join(result_dir, "imgs"),
        logger,
    )

    result = {}
    result_file_name = "results.json"
    baseline_result = None
    if baseline_dir:
        result_file_path = os.path.join(baseline_dir, result_file_name)
        if os.path.exists(result_file_path):
            with open(result_file_path, "r") as fp:
                baseline_result = json.load(fp)

    with comfy_client_context(port=comfy_port) as client:
        logger.info(f"Testing workflows: {workflow}")

        for i, comfy_graph in enumerate(dispatch_generator(workflow)):
            start_time = time.time()
            images = client.get_images(comfy_graph.graph)
            end_time = time.time()

            if not images:
                logger.error("No images generated")
                raise ValueError("No images generated")
            if len(images) != 1:
                logger.error(
                    f"Expected 1 image, but got {len(images)} images. Batch Size == 1"
                )
                raise ValueError(
                    f"Expected 1 image, but got {len(images)} images. Batch Size == 1"
                )

            for images_output in images.values():
                for image_data in images_output:
                    out = processor.process_image(image_data, i)
                    result[i] = {
                        "height": out.height,
                        "width": out.width,
                        "image_path": out.image_path,
                    }
                    e2e_time = end_time - start_time
                    logger.info(f"Workflow {i} E2E:  {e2e_time:.2f} seconds")
                    result[i].update({"e2e_time": e2e_time})

                    if baseline_dir:
                        baseline_image_path = os.path.join(
                            baseline_dir, "imgs", f"image_{i}.png"
                        )
                        baseline_image = Image.open(baseline_image_path)
                        pil_image = Image.open(BytesIO(image_data))
                        ssim_value = calculate_ssim(pil_image, baseline_image)
                        result[i].update(
                            {
                                "ssim_value": ssim_value,
                                "basic_image_path": baseline_image_path,
                            }
                        )
                        logger.info(f"SSIM: {ssim_value=}")
                        # assert (
                        #     ssim_value > ssim_threshold
                        # ), f"SSIM value {ssim_value} is not greater than the threshold {ssim_threshold}"

                        if baseline_result:
                            basic_time = baseline_result[str(i)]["e2e_time"]
                            result[i].update({"basic_e2e_time": basic_time})
                            percentage_improvement = (
                                basic_time - e2e_time
                            ) / basic_time
                            result[i].update(
                                {"percentage_improvement": percentage_improvement}
                            )

        # Save results to a JSON file
        result_file = os.path.join(result_dir, result_file_name)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=4)

        logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    args = parse_args()
    run_workflow(
        args.workflow,
        args.comfy_port,
        args.output_images,
        args.baseline_dir,
        args.ssim_threshold,
        args.output_dir,
        args.exp_name,
    )
