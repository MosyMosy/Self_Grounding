import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image, ImageOps


def depth_image_to_pointcloud(depth, scale, K):
    """
    Convert a depth image to a point cloud in 3D space.

    Args:
        depth (torch.Tensor): Depth image tensor of shape (B, H, W).
        scale (torch.Tensor): Scale factor tensor of shape (B,).
        K (torch.Tensor): Camera intrinsic matrix tensor of shape (B, 3, 3).

    Returns:
        torch.Tensor: Point cloud tensor of shape (B, H, W, 3), where the last dimension represents (X, Y, Z) coordinates.
    """
    assert len(depth.shape) == 3
    assert len(K.shape) == 3
    assert len(scale.shape) == 1
    assert depth.shape[0] == K.shape[0] == scale.shape[0]

    B, H, W = depth.shape
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    # Generate pixel coordinates grid
    # !!!!!!!!!!!!!! Much faster than using torch.meshgrid !!!!!!!!!!!!!!
    u = torch.arange(W, device=depth.device).unsqueeze(0).unsqueeze(0)
    v = torch.arange(H, device=depth.device).unsqueeze(1).unsqueeze(0)
    u = u.float().expand(B, H, -1)
    v = v.float().expand(B, -1, W)

    ###### Homoheneous calculation is not memory efficient
    ###### Here I use the gridwise calculation
    """
        z = d / depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
    """
    Z = depth * scale / 1000
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    xyz = torch.stack((X, Y, Z), dim=-1)
    return xyz


# this function is copied from https://github.com/JiehongLin/SAM-6D/
class CropResizePad:
    def __init__(self, target_size):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.target_ratio = self.target_size[1] / self.target_size[0]
        self.target_h, self.target_w = target_size
        self.target_max = max(self.target_h, self.target_w)

    def __call__(self, images, boxes):
        box_sizes = boxes[:, 2:] - boxes[:, :2]
        scale_factor = self.target_max / torch.max(box_sizes, dim=-1)[0]
        processed_images = []
        for image, box, scale in zip(images, boxes, scale_factor):
            # crop and scale
            image = image[:, box[1] : box[3], box[0] : box[2]]
            image = F.interpolate(image.unsqueeze(0), scale_factor=scale.item())[0]
            # pad and resize
            original_h, original_w = image.shape[1:]
            original_ratio = original_w / original_h

            # check if the original and final aspect ratios are the same within a margin
            if self.target_ratio != original_ratio:
                padding_top = max((self.target_h - original_h) // 2, 0)
                padding_bottom = self.target_h - original_h - padding_top
                padding_left = max((self.target_w - original_w) // 2, 0)
                padding_right = self.target_w - original_w - padding_left
                image = F.pad(
                    image, (padding_left, padding_right, padding_top, padding_bottom)
                )
            assert (
                image.shape[1] == image.shape[2]
            ), f"image {image.shape} is not square after padding"

            image = F.interpolate(
                image.unsqueeze(0), scale_factor=self.target_h / image.shape[1]
            )[0]
            processed_images.append(image)
        return torch.stack(processed_images)


class CropResizePad_v2:
    def __init__(self, target_size):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.target_ratio = self.target_size[1] / self.target_size[0]
        self.target_h, self.target_w = target_size
        self.target_max = max(self.target_h, self.target_w)

    def __call__(self, images, boxes=None):
        num_images = images.size(0)

        if boxes is None:
            # Create boxes that span the entire image
            num_images = images.shape[0]
            _, _, h, w = images.shape
            boxes = torch.tensor([[0, 0, w, h]] * num_images)

        # Calculate box sizes and scale factors
        box_sizes = boxes[:, 2:] - boxes[:, :2]
        scale_factors = self.target_max / torch.max(box_sizes, dim=-1)[0]

        cropped_images = []
        padding_info = []
        for i in range(num_images):
            image, box = images[i], boxes[i]
            cropped_image = image[
                :, int(box[1]) : int(box[3]), int(box[0]) : int(box[2])
            ]
            cropped_h, cropped_w = cropped_image.shape[-2:]
            if cropped_h == 0 or cropped_w == 0:
                cropped_images.append(
                    torch.zeros(image.shape[0], self.target_h, self.target_w).to(
                        image.device
                    )
                )
                padding_info.append((0, 0, 1))
                continue
            scaled_image = F.interpolate(
                cropped_image.unsqueeze(0), scale_factor=scale_factors[i].item()
            )[0]

            original_h, original_w = scaled_image.shape[1:]
            padding_bottom = max(self.target_h - original_h, 0)
            padding_right = max(self.target_w - original_w, 0)

            padded_image = F.pad(scaled_image, (0, padding_right, 0, padding_bottom))

            final_image = F.interpolate(
                padded_image.unsqueeze(0), size=(self.target_h, self.target_w)
            )[0]
            cropped_images.append(final_image)

            padding_info.append((padding_bottom, padding_right, scale_factors[i]))

        return torch.stack(cropped_images)

    def reverse(self, images, original_sizes):
        original_h, original_w = original_sizes
        target_h, target_w = images.shape[-2:]

        if images.shape[1] == 0:
            return torch.zeros(images.shape[0], 1, original_h, original_w).to(
                images.device
            )

        # Calculate padding info
        padding_bottom = max(target_h - original_h, 0)
        padding_right = max(target_w - original_w, 0)

        # Reverse padding
        unpadded_image = images[
            :, :, : target_h - padding_bottom, : target_w - padding_right
        ]

        unscaled_image = F.interpolate(
            unpadded_image,
            size=(original_h, original_w),
        )

        return unscaled_image


def equalize_brightness(image: Image.Image):
    # Convert the image to HSV
    hsv_image = image.convert("HSV")

    # Split into H, S, V channels
    h, s, v = hsv_image.split()

    # Equalize the Value (Brightness) channel
    v_equalized = ImageOps.equalize(v)

    # Merge the channels back
    hsv_equalized = Image.merge("HSV", (h, s, v_equalized))

    # Convert back to RGB and save the result
    rgb_equalized = hsv_equalized.convert("RGB")
    return rgb_equalized


def constant_brightness(image: Image.Image):
    # Convert the image to HSV
    hsv_image = image.convert("HSV")

    # Split into H, S, V channels
    h, s, v = hsv_image.split()

    v_constant = Image.new("L", v.size, 128)

    # Merge the channels back
    hsv_v_constant = Image.merge("HSV", (h, s, v_constant))

    # Convert back to RGB and save the result
    rgb_v_constant = hsv_v_constant.convert("RGB")
    return rgb_v_constant
