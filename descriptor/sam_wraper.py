import warnings
import torch

# from segment_anything import SamPredictor, sam_model_registry
from torchvision import transforms
from dataset.util import CropResizePad_v2
from descriptor.descriptor_base import Descriptor_Base
from segment_anything import SamPredictor, sam_model_registry

model_types = ["vit_h", "vit_l", "vit_b"]
channel_sizes = {
    "vit_h": 256,
    "vit_l": 256,
    "vit_b": 256,
}

checkpoint_dic = {
    "vit_h": "checkpoints/SAM_Wraper/sam_vit_h_4b8939.pth",
    "vit_l": "checkpoints/SAM_Wraper/sam_vit_l_0b3195.pth",
    "vit_b": "checkpoints/SAM_Wraper/sam_vit_b_01ec64.pth",
}


class Sam_Wraper(Descriptor_Base):
    def __init__(
        self,
        model_type="vitg14_reg",
        device="cpu",
        obj_point_size=1024,
        num_templates_per_obj=32,
        input_size=(768, 1024),  # h*w
        scaled_input_size=(1024, 1024),
        output_spatial_size=(48, 64),
        scaled_output_spacial_size=(64, 64),
        original_image_size=(540, 720),
    ) -> None:
        assert model_type in model_types, f"model_type should be one of {model_types}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sam = sam_model_registry[model_type](
                checkpoint=checkpoint_dic[model_type],
            )
        self.sam.eval().to(device)
        self.predictor = SamPredictor(self.sam)
        self.sam.image_encoder.img_size = input_size[1]

        self.input_size = input_size
        self.scaled_input_size = scaled_input_size
        self.output_spatial_size = output_spatial_size
        self.scaled_output_spacial_size = scaled_output_spacial_size
        self.output_channels = channel_sizes[model_type]
        self.patch_size = 16
        self.original_image_size = original_image_size
        # template settings
        self.obj_point_size = obj_point_size
        self.num_templates_per_obj = num_templates_per_obj

        # identity transform
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)]
        )
        self.inv_trans = transforms.Lambda(lambda x: x / 255)  # Identity transform

        self.cropper = CropResizePad_v2(self.input_size)
        self.scaled_cropper = CropResizePad_v2(self.scaled_input_size)

        self.name = f"SAM_{model_type}_obj_point_size_{self.obj_point_size}_num_temp_{self.num_templates_per_obj}"

    def encode_image_base(
        self,
        processed_image: torch.Tensor,
        scaled: bool = False,
    ):
        if len(processed_image) > 1:
            features = []
            for i in range(len(processed_image)):
                with torch.no_grad():
                    self.predictor.set_torch_image(
                        processed_image[i].unsqueeze(0),
                        original_image_size=self.original_image_size,
                    )
                embedding = self.predictor.get_image_embedding()
                features.append(embedding)
            features = torch.cat(features, dim=0)
        else:
            with torch.no_grad():
                self.predictor.set_torch_image(
                    processed_image,
                    original_image_size=self.original_image_size,
                )
            features = self.predictor.get_image_embedding()

        if scaled:
            features = features[
                :,
                :,
                : self.scaled_output_spacial_size[0],
                : self.scaled_output_spacial_size[1],
            ]
        else:
            features = features[
                :, :, : self.output_spatial_size[0], : self.output_spatial_size[1]
            ]
        features = features.permute(0, 2, 3, 1).reshape(
            features.shape[0], -1, self.output_channels
        )
        features /= torch.norm(features, dim=-1, keepdim=True)
        return features
