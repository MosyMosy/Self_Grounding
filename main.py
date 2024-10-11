import os
from time import time

# from segmentation.sam2 import SAM2_Wraper
import torch
import torch.utils
from torchvision import utils as tvutils
from torchvision.ops import batched_nms
from tqdm import tqdm
import numpy as np
import json
from torchvision import utils as vutils

from dataset.bop import BOP_dataset, BOP_dataset_info, BOP_obj_template_dataset
from descriptor.dino_wraper import Dino_Descriptors, Dino_Wraper
from descriptor.sam_wraper import Sam_Wraper

from pipeline_util import pipeline_step, seed_all


dataset_name = "tless"
do_intra_inter_scores = False
test = False


dataset_settings = {
    "root_dir": f"/home/epi/Moslem/dataset/bop/{dataset_name}",  # "/export/datasets/public/3d/BOP/T-LESS",
    "dataset_info": BOP_dataset_info[dataset_name],
    "depth_name_template": BOP_dataset_info[dataset_name]["depth_name_template"],
}


train_dataset_settings = {
    "mode": "train",
    "split": BOP_dataset_info[dataset_name]["train_dir"],
    "get_obj_gt_visible_masks": False,
    "image_name_template": BOP_dataset_info[dataset_name]["image_name_template"],
}

test_dataset_settings = {
    "mode": "test",
    "split": BOP_dataset_info[dataset_name]["test_dir"],
    "get_obj_gt_visible_masks": False,
    "image_name_template": BOP_dataset_info[dataset_name][
        "image_name_template"
    ].replace("jpg", "png"),
    "get_obj_gt_visible_masks": True,
}

# descriptor_settings_dino = {
#     "class": Dino_Wraper,
#     "model_type": "vitl14_reg",
#     "obj_point_size": 1024,
#     "num_templates_per_obj": 32,
# }

descriptor_settings_dino_descriptor = {
    "class": Dino_Descriptors,
    "model_type": "vitl14_reg",
    "obj_point_size": 1024,
    "num_templates_per_obj": 120,
}

descriptor_settings_sam = {
    "class": Sam_Wraper,
    "model_type": "vit_h",
    "obj_point_size": 1024,
    "num_templates_per_obj": 32,
}


descriptor_setting = descriptor_settings_dino_descriptor

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run_pipeline(device):
    with pipeline_step("Preparing the environment"):
        pipeline_step.print_full_width(
            f"initializing the descriptor: {descriptor_setting['class'].__name__} {descriptor_setting['model_type']}"
        )
        descriptor = descriptor_setting["class"](
            descriptor_setting["model_type"],
            device,
            descriptor_setting["obj_point_size"],
            descriptor_setting["num_templates_per_obj"],
        )

        pipeline_step.print_full_width(
            f"loading from the test split of {dataset_name} dataset"
        )
        test_ds = BOP_dataset(
            **dataset_settings,
            **test_dataset_settings,
            transform_image=descriptor.transform,
            # cropper=descriptor.cropper,
            enhance_rgb=False,
        )
        test_ds.prepare_metadata()
        test_dl = test_ds.get_dataloader(batch_size=1, num_workers=4, shuffle=True)
        template_path = f"offline_data/{descriptor.name}/{dataset_name}_templates.pth"

    with pipeline_step("Offline preparation"):
        if os.path.exists(template_path):
            pipeline_step.print_full_width("Loading the object's templates features")
            template_dic = torch.load(template_path)
            obj_templates_average_feats = template_dic["obj_templates_average_feats"]

            scaled_obj_templates_feats = template_dic["scaled_obj_templates_feats"]
            scaled_obj_templates_average_feats = template_dic[
                "scaled_obj_templates_average_feats"
            ]
            obj_patched_max_size = template_dic["obj_patched_max_size"]
        else:
            with pipeline_step(
                "Adding object's templates features to the object's point cloud"
            ):
                pipeline_step.print_full_width(
                    f"loading from the train split of {dataset_name} dataset"
                )
                obj_template_ds = BOP_obj_template_dataset(
                    **dataset_settings,
                    **train_dataset_settings,
                    transform_image=descriptor.transform,
                    visible_fract_threshold=0.5,
                    num_templates=descriptor.num_templates_per_obj,
                    # cropper=descriptor.cropper,  # CropResizePad(224)
                )
                obj_template_dl = obj_template_ds.get_dataloader(
                    batch_size=1, num_workers=2, shuffle=False
                )
                obj_templates_average_feats = []
                scaled_obj_templates_feats = []
                scaled_obj_templates_average_feats = []
                obj_patched_max_size = []
                for sample in tqdm(
                    obj_template_dl, desc="Adding features to the point cloud"
                ):
                    image = sample["image"][0].to(device)
                    masks = sample["mask"][0].to(device)
                    xyzs = sample["xyz"][0].to(device)

                    obj_pc = obj_template_ds.get_obj_pc(
                        model_id=sample["obj_id"][0].item(),
                        pc_size=descriptor.obj_point_size * 3,
                    ).to(device)
                    gt_R, gt_t = (
                        sample["rotation"][0].to(device),
                        sample["translation"][0].to(device),
                    )

                    (
                        template_point_feat,
                        obj_templates_average_feat,
                        scaled_template_point_feat,
                        scaled_obj_templates_feat,
                        scaled_obj_templates_average_feat,
                        patched_max_size,
                    ) = extract_combined_point_features(
                        image, masks, xyzs, gt_R, gt_t, obj_pc, descriptor
                    )

                    obj_templates_average_feats.append(obj_templates_average_feat)
                    scaled_obj_templates_feats.append(scaled_obj_templates_feat)
                    scaled_obj_templates_average_feats.append(
                        scaled_obj_templates_average_feat
                    )
                    obj_patched_max_size.append(patched_max_size)
                obj_templates_average_feats = torch.stack(obj_templates_average_feats)

                scaled_obj_templates_feats = torch.stack(scaled_obj_templates_feats)
                scaled_obj_templates_average_feats = torch.stack(
                    scaled_obj_templates_average_feats
                )
                obj_patched_max_size = torch.stack(obj_patched_max_size)
                sample = None

                # saving the object's templates features
                pipeline_step.print_full_width("Saving the object's templates features")
                os.makedirs(os.path.dirname(template_path), exist_ok=True)
                template_dic = {}
                template_dic["obj_templates_average_feats"] = (
                    obj_templates_average_feats
                )
                template_dic["scaled_obj_templates_feats"] = scaled_obj_templates_feats
                template_dic["scaled_obj_templates_average_feats"] = (
                    scaled_obj_templates_average_feats
                )
                template_dic["obj_patched_max_size"] = obj_patched_max_size

                torch.save(template_dic, template_path)


if __name__ == "__main__":
    seed_all(41)
    run_pipeline(device)
