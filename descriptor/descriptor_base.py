import torch
import warnings

# from segment_anything import SamPredictor, sam_model_registry
from torchvision import transforms
from dataset.util import CropResizePad_v2


model_types = ["vits14_reg", "vitb14_reg", "vitl14_reg", "vitg14_reg"]
channel_sizes = {
    "vits14_reg": 384,
    "vitb14_reg": 768,
    "vitl14_reg": 1024,
    "vitg14_reg": 1536,
}


class Descriptor_Base:
    def __init__(
        self,
        model_type="vitg14_reg",
        device="cpu",
        obj_point_size=1024,
        num_templates_per_obj=32,
        input_size=(532, 714),  # h*w
        scaled_input_size=(224, 224),
        output_spatial_size=(38, 51),
        scaled_output_spacial_size=(16, 16),
        original_image_size=(540, 720),
    ) -> None:
        assert model_type in model_types, f"model_type should be one of {model_types}"
        with warnings.catch_warnings():
            self.model = torch.hub.load(
                "facebookresearch/dinov2", f"dinov2_{model_type}"
            )
        self.model.eval()
        self.model.to(device)

        self.input_size = input_size
        self.scaled_input_size = scaled_input_size
        self.output_spatial_size = output_spatial_size
        self.scaled_output_spacial_size = scaled_output_spacial_size
        self.output_channels = channel_sizes[model_type]
        self.patch_size = 14
        self.original_image_size = original_image_size
        # template settings
        self.obj_point_size = obj_point_size
        self.num_templates_per_obj = num_templates_per_obj

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.inv_trans = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )

        self.cropper = CropResizePad_v2(self.input_size)
        self.scaled_cropper = CropResizePad_v2(self.scaled_input_size, square_bbox=True)

        self.name = f"Dino_{model_type}_obj_point_size_{self.obj_point_size}_num_temp_{self.num_templates_per_obj}"

    def encode_image_base(self, image: torch.Tensor, scaled: bool = False):
        raise NotImplementedError

    def encode_image_with_rotation(
        self, image: torch.Tensor, inplane_rotation=False, scaled=False
    ):
        with torch.no_grad():
            B_in = image.shape[0]
            if inplane_rotation:
                # Concatenate the original and rotated images
                image = torch.cat(
                    [
                        image,
                        torch.rot90(image, k=1, dims=[2, 3]),
                        torch.rot90(image, k=2, dims=[2, 3]),
                        torch.rot90(image, k=3, dims=[2, 3]),
                    ],
                    dim=0,
                )
                features = self.encode_image_base(image, scaled=scaled)
                features = features.flatten(0, 1)
                B_feat = features.shape[0]
                features = features.permute(0, 2, 1).view(
                    B_feat * 4,
                    self.output_channels,
                    *self.scaled_output_spacial_size,
                )
                features = (
                    features[:B_feat]
                    + torch.rot90(
                        features[1 * B_feat : 2 * B_feat],
                        k=3,
                        dims=[2, 3],
                    )
                    + torch.rot90(
                        features[2 * B_feat : 3 * B_feat],
                        k=2,
                        dims=[2, 3],
                    )
                    + torch.rot90(
                        features[3 * B_feat : 4 * B_feat],
                        k=1,
                        dims=[2, 3],
                    )
                ) / 4
                C = features.shape[1]
                features = features.permute(0, 2, 3, 1).view(B_feat, -1, C)
                features = features.view(B_in, B_feat // B_in, -1, C).mean(dim=1)
            else:
                features = self.encode_image_base(image, scaled=scaled)
        return features

    def encode_image(
        self, image: torch.Tensor, inplane_rotation=False, mask=None, is_scaled=False
    ):
        if (not is_scaled) and (
            image.shape[-1] != self.input_size[-1]
            or image.shape[-2] != self.input_size[-2]
        ):
            # image = transforms.CenterCrop(540)(image)
            image_sized = self.cropper(image)
        else:
            image_sized = image

        features = self.encode_image_with_rotation(
            image_sized, inplane_rotation=inplane_rotation
        )
        features = features.permute(0, 1, 3, 2)
        spatial_size = (
            self.scaled_output_spacial_size if is_scaled else self.output_spatial_size
        )
        features = features.view(*features.shape[:3], *spatial_size)
        if mask is not None:
            masks_patched = (
                torch.functional.F.interpolate(
                    mask.float(),
                    size=spatial_size,
                    mode="bilinear",
                )
                > 0.5
            )
            features_average = (features * masks_patched.unsqueeze(1)).sum(
                dim=(3, 4)
            ) / (masks_patched.sum() + 1e-8)
            return features, features_average, masks_patched
        else:
            return features

    # def encode_image_scaled_masked(self, image, mask, bbox, inplane_rotation=False):
    #     assert len(mask.shape) == 4, "mask should be batched"
    #     assert mask.shape[1] == 1, "mask should be binary"
    #     assert (
    #         mask.shape[0] == bbox.shape[0]
    #     ), "mask and bbox should have same batch size"
    #     assert (
    #         mask.shape[0] == bbox.shape[0]
    #     ), "mask and bbox should have same batch size"

    #     B = mask.shape[0]
    #     test_image_cropped = self.scaled_cropper(image, bbox)
    #     masks_cropped = self.scaled_cropper(mask.float(), bbox)

    #     # test_image_cropped = test_image_cropped * gt_masks_cropped # This is not good
    #     test_image_cropped_embedding = self.encode_image_with_rotation(
    #         test_image_cropped, inplane_rotation=inplane_rotation, scaled=True
    #     )
    #     test_image_cropped_embedding = test_image_cropped_embedding.permute(0, 1, 3, 2)
    #     test_image_cropped_embedding = test_image_cropped_embedding.view(
    #         *test_image_cropped_embedding.shape[:3],
    #         *self.scaled_output_spacial_size,
    #     )

    #     masks_cropped_patched = (
    #         torch.functional.F.interpolate(
    #             masks_cropped.float(),
    #             size=self.scaled_output_spacial_size,
    #             mode="bilinear",
    #         )
    #         > 0
    #     )

    #     test_image_cropped_embedding *= masks_cropped_patched.unsqueeze(1)
    #     # test_image_cropped_embedding = test_image_cropped_embedding.permute(
    #     #     0, 1, 3, 4, 2
    #     # )

    #     # Average the embeddings
    #     masks_patched_lengths = masks_cropped_patched.sum(dim=(2, 3))
    #     test_embedding_masked_mean = test_image_cropped_embedding.sum(
    #         dim=(3, 4)
    #     ) / masks_patched_lengths.unsqueeze(-1)

    #     return (
    #         test_image_cropped_embedding,
    #         masks_cropped_patched,
    #         test_embedding_masked_mean,
    #     )

    def to_org_size(self, prompt_map, H_org, W_org):
        prompt_map = torch.nn.functional.interpolate(
            prompt_map.unsqueeze(0),
            size=(H_org, W_org),
            mode="nearest",
            # align_corners=False,
        )
        prompt_map = prompt_map[:, :, :H_org, :W_org]
        return prompt_map
