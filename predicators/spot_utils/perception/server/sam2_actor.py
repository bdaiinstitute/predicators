# This implimentation was done using the ultralytics library instead of the official SAM2 library since the ultralytics
# library is a little more user friendly as well as works on older python versions such as 3.9. The official SAM2 library
# requires python 3.10 and above. The library supports the most recent SAM2.1 models and if able to do point, bbox, or
# complete segmentation. If needed exploring the official SAM2 library could be a future TODO

# pip install ultralytics

import logging
import os

import numpy as np
import ray
import requests
import torch
from PIL import Image
from ultralytics import SAM

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class SAM2Actor:
    def __init__(self, model_name="sam2.1_l.pt", use_gpu=False):
        # Model_names:

        # SAM 2 tiny	sam2_t.pt
        # SAM 2 small	sam2_s.pt
        # SAM 2 base	sam2_b.pt
        # SAM 2 large	sam2_l.pt

        # SAM 2.1 tiny	sam2.1_t.pt
        # SAM 2.1 small	sam2.1_s.pt
        # SAM 2.1 base	sam2.1_b.pt
        # SAM 2.1 large	sam2.1_l.pt

        super().__init__(use_gpu)
        self.device = "cuda" if use_gpu else "cpu"
        self.model = SAM(model_name).to(self.device)
        print("Is SAM2:", self.model.is_sam2)

    def get_device_info(self):
        return f"Model: {next(self.model.parameters()).device}, Input tensors: {self.device}"

    def predict(self, image, input=None, type="point"):
        """
        Args:
            image: PIL Image
            input: For type="point", should be list of [x,y] coordinates
            type: "point" or "box"
        """
        try:
            # Convert input points to correct format
            if type == "point" and input is not None:
                # Ensure input is numpy array with correct shape
                points = np.array(input)
                if len(points) == 0:
                    points = np.zeros((0, 2))  # Empty array with correct shape
                else:
                    points = points.reshape(-1, 2)  # Ensure shape is (N, 2)

                # Create labels array (1 for each point)
                labels = np.ones(len(points))

                # Convert to torch tensors with batch dimension
                points = torch.as_tensor(
                    points, dtype=torch.float32, device=self.device
                ).unsqueeze(
                    0
                )  # Add batch dim
                labels = torch.as_tensor(
                    labels, dtype=torch.int64, device=self.device
                ).unsqueeze(
                    0
                )  # Add batch dim

                # Pass points and labels directly to predict
                results = self.model.predict(
                    image,
                    points=points,  # Pass points tensor with batch dim
                    labels=labels,  # Pass labels tensor with batch dim
                    device=self.device,
                    retina_masks=True,  # Enable high-res masks
                    conf=0.5,  # Confidence threshold
                    iou=0.9,  # NMS IoU threshold
                )
            elif type == "box":
                # Convert boxes to tensor with batch dimension if provided
                if input is not None:
                    boxes = torch.as_tensor(
                        input, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                else:
                    boxes = None

                results = self.model.predict(
                    image,
                    boxes=boxes,
                    device=self.device,
                    retina_masks=True,
                    conf=0.5,
                    iou=0.9,
                )
            else:
                results = self.model.predict(
                    image,
                    device=self.device,
                    retina_masks=True,
                    conf=0.5,
                    iou=0.9,
                )

            # Return first result and ensure it's on CPU for serialization
            result = results[0]
            if hasattr(result, "masks") and result.masks is not None:
                result.masks = result.masks.cpu()
            if hasattr(result, "boxes") and result.boxes is not None:
                result.boxes = result.boxes.cpu()

            return result

        except Exception as e:
            print(f"Error in SAM2Actor.predict: {str(e)}")
            import traceback

            traceback.print_exc()
            raise


if __name__ == "__main__":
    # Start a Ray cluster
    ray.init()

    print(ray.available_resources())

    # Adjust based on your setup
    use_gpu = torch.cuda.is_available()

    actor_options = {"num_gpus": 1} if use_gpu else {}

    actor = SAM2Actor.options(**actor_options).remote(use_gpu=use_gpu)
    logging.info("Actor ready for inference.")

    url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"  # noqa
    image = Image.open(requests.get(url, stream=True).raw)

    # Create tmp directory for outputs
    tmp_dir = os.path.join("src", "models", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    result = ray.get(actor.predict.remote(image))
    result2 = ray.get(actor.predict.remote(image, input=[[450, 600]], type="point"))
    logging.info("Inference complete.")

    # Save results instead of showing
    result_path = os.path.join(tmp_dir, "sam_full.png")
    result2_path = os.path.join(tmp_dir, "sam_point.png")
    result.save(result_path)
    result2.save(result2_path)
    print(f"Results saved to: {result_path} and {result2_path}")

    ray.shutdown()
