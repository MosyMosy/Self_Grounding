import torch
import os
from time import time
import random
import numpy as np


class pipeline_step:
    def __init__(self, name: str, speak: bool = True):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.speak = speak

    def __enter__(self):
        self.start_time = time()
        if self.speak:
            pipeline_step.print_full_width("-", "-")
            pipeline_step.print_full_width(" ", " ")
            pipeline_step.print_full_width("=", "=")
            opening_message = f"Starting {self.name}"
            pipeline_step.print_full_width(opening_message, "")
            pipeline_step.print_full_width(". " * (len(opening_message) // 2), " ")

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
        self.end_time = time()
        self.duration = self.end_time - self.start_time
        if self.speak:
            closing_message = f"{self.name} completed in {self.duration:.2f} seconds."
            pipeline_step.print_full_width(". " * (len(closing_message) // 2), "")
            pipeline_step.print_full_width(closing_message, "")
            pipeline_step.print_full_width("-", "-")
        return False

    @staticmethod
    def print_full_width(text: str, filler_char: str = " ") -> None:
        terminal_width = os.get_terminal_size().columns
        print(f"{text:{filler_char}^{terminal_width}}")


def seed_all(seed):
    # Set seed for Python random number generator
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch CPU
    torch.manual_seed(seed)

    # Set seed for PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_features(image, masks, descriptor, inplane_rotation=False):
    image_sized = descriptor.cropper(image)
    masks_sized = descriptor.cropper(masks)

    B = image_sized.shape[0]
    # Extract embeddings and process normal features

    # Extract scaled features
    (
        normal_embeddings,
        normal_embeddings_averages,
        scaled_embeddings,
        scaled_embeddings_average,
    ) = ([], [], [], [])
    for i in range(masks_sized.shape[0]):
        normal_embedding, normal_embedding_average, patched_mask = (
            descriptor.encode_image(
                image_sized[i].unsqueeze(0), mask=masks_sized[i].unsqueeze(0)
            )
        )
        normal_embedding, normal_embedding_average = (
            normal_embedding[0],
            normal_embedding_average[0],
        )

        bbox = get_bounding_boxes_batch(masks_sized[i].unsqueeze(0).squeeze(1))
        scaled_embedding, _, scaled_embedding_average = (
            descriptor.encode_image_scaled_masked(
                image_sized[i],
                masks_sized[i].unsqueeze(0),
                bbox,
                inplane_rotation=inplane_rotation,
            )
        )

        normal_embeddings.append(normal_embedding)
        normal_embeddings_averages.append(normal_embedding_average)
        scaled_embeddings.append(scaled_embedding.squeeze(0))
        scaled_embeddings_average.append(scaled_embedding_average.squeeze(0))

    normal_embeddings = torch.stack(normal_embeddings)
    normal_embeddings_averages = torch.stack(normal_embeddings_averages)
    scaled_embeddings = torch.stack(scaled_embeddings).permute(0, 3, 1, 2)
    scaled_embeddings_average = torch.stack(scaled_embeddings_average)

    return (
        normal_embeddings,
        normal_embeddings_averages,
        scaled_embeddings,
        scaled_embeddings_average,
    )


def get_bounding_box(mask):
    """
    Get the bounding box coordinates of a single segmentation mask.

    Parameters:
    mask (torch.Tensor): A binary segmentation mask of shape (H, W)

    Returns:
    bbox (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max)
    """
    # Find the indices of non-zero elements
    non_zero_indices = torch.nonzero(mask)

    if non_zero_indices.numel() == 0:
        return torch.tensor(
            [0, 0, 2, 2], device=mask.device
        )  # If the mask is empty, return a zero-size bounding box

    # Get the min and max coordinates
    y_min, x_min = torch.min(non_zero_indices, dim=0)[0]
    y_max, x_max = torch.max(non_zero_indices, dim=0)[0]

    return torch.stack([x_min, y_min, x_max, y_max])


def get_bounding_boxes_batch(masks):
    """
    Get bounding box coordinates for a batch of segmentation masks.

    Parameters:
    masks (torch.Tensor): A batch of binary segmentation masks of shape (B, H, W)

    Returns:
    bboxes: tensor of bounding box coordinates for each mask
    """
    bboxes = []
    for mask in masks:
        bbox = get_bounding_box(mask)
        bboxes.append(bbox)

    return torch.stack(bboxes)
