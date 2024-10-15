import torch
import warnings

# from segment_anything import SamPredictor, sam_model_registry
from torchvision import transforms
from descriptor.vit_extractor import ViTExtractor
from dataset.util import CropResizePad_v2
import torchvision.utils as tvutil
from descriptor.descriptor_base import Descriptor_Base

import warnings

import torch


depths = {
    "vits14_reg": 12,
    "vitb14_reg": 12,
    "vitl14_reg": 24,
    "vitg14_reg": 40,
}


class Dino_Wraper(Descriptor_Base):
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
    ) -> None:
        super().__init__(
            model_type,
            device,
            obj_point_size,
            num_templates_per_obj,
            input_size=input_size,
            scaled_input_size=scaled_input_size,
            output_spatial_size=output_spatial_size,
            scaled_output_spacial_size=scaled_output_spacial_size,
        )
        self.name = f"Dino_{model_type}_obj_point_size_{self.obj_point_size}_num_temp_{self.num_templates_per_obj}"

    def encode_image_base(
        self, image: torch.Tensor, scaled: bool = False, g_info: torch.Tensor = None
    ):
        features = self.model.get_intermediate_layers(image, g_info=g_info)[0]
        features /= torch.norm(features, dim=-1, keepdim=True)
        return features


class Dino_Descriptors(Dino_Wraper):
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
        stride=(14, 14),
    ) -> None:
        super().__init__(
            model_type,
            device,
            obj_point_size,
            num_templates_per_obj,
            input_size=input_size,
            scaled_input_size=scaled_input_size,
            output_spatial_size=output_spatial_size,
            scaled_output_spacial_size=scaled_output_spacial_size,
        )

        self.depth = depths[model_type]
        self.extractor = ViTExtractor(self.model, stride=stride)
        self.name = f"Dino-Des_{model_type}_obj_point_size_{self.obj_point_size}_num_temp_{self.num_templates_per_obj}"

    def set_resolution(self, stride):
        self.model = self.extractor.patch_vit_resolution(self.model, stride)
        patch_division_H = self.patch_size // stride[0]
        patch_division_W = self.patch_size // stride[1]
        self.output_spatial_size = (
            self.input_size[0] // (self.patch_size // patch_division_H)
            - (patch_division_H - 1),
            self.input_size[1] // (self.patch_size // patch_division_W)
            - (patch_division_W - 1),
        )

    def encode_image_base(
        self, image: torch.Tensor, scaled: bool = False, g_info: torch.Tensor = None
    ):
        assert (
            len(g_info) == self.depth
        ), f"g_info should have {self.depth} elements (Network's layers)"
        layers = [i for i in range(self.depth)]
        features = self.extractor.extract_descriptors(
            image,
            g_info=g_info,
            layer_idx=layers,
            register_size=self.model.num_register_tokens,
        )
        features = torch.stack(
            [features["query"], features["key"], features["value"], features["token"]],
            dim=2,
        )
        return features
