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

from pipeline_util import pipeline_step, seed_all, extract_features


dataset_name = "tless"
do_intra_inter_scores = False
test = False


dataset_settings = {
    "root_dir": f"/export/datasets/public/3d/BOP/{dataset_name}",  # "/export/datasets/public/3d/BOP/T-LESS",
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
    "num_templates_per_obj": 16,
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
            obj_templates_feats = template_dic["obj_templates_feats"]
            obj_templates_average_feats = template_dic["obj_templates_average_feats"]

            obj_templates_feats_scaled = template_dic["obj_templates_feats_scaled"]
            obj_templates_average_feats_scaled = template_dic[
                "obj_templates_average_feats_scaled"
            ]
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

                obj_templates_feats = []
                obj_templates_average_feats = []
                obj_templates_feats_scaled = []
                obj_templates_average_feats_scaled = []
                for sample in tqdm(
                    obj_template_dl, desc="Adding features to the point cloud"
                ):
                    image = sample["image"][0].to(device)
                    masks = sample["mask"][0].to(device)

                    (
                        obj_templates_feat,
                        obj_templates_average_feat,
                        scaled_obj_templates_feat,
                        scaled_obj_templates_average_feat,
                    ) = extract_features(
                        image, masks, descriptor, inplane_rotation=False, batch_size=1
                    )

                    obj_templates_feats.append(obj_templates_feat)
                    obj_templates_average_feats.append(obj_templates_average_feat)
                    obj_templates_feats_scaled.append(scaled_obj_templates_feat)
                    obj_templates_average_feats_scaled.append(
                        scaled_obj_templates_average_feat
                    )
                obj_templates_feats = torch.stack(obj_templates_feats)
                obj_templates_average_feats = torch.stack(obj_templates_average_feats)

                obj_templates_feats_scaled = torch.stack(obj_templates_feats_scaled)
                obj_templates_average_feats_scaled = torch.stack(
                    obj_templates_average_feats_scaled
                )
                sample = None

                # saving the object's templates features
                pipeline_step.print_full_width("Saving the object's templates features")
                os.makedirs(os.path.dirname(template_path), exist_ok=True)

                template_dic = {}
                template_dic["obj_templates_feats"] = obj_templates_feats
                template_dic["obj_templates_average_feats"] = (
                    obj_templates_average_feats
                )
                template_dic["obj_templates_feats_scaled"] = obj_templates_feats_scaled
                template_dic["obj_templates_average_feats_scaled"] = (
                    obj_templates_average_feats_scaled
                )

                torch.save(template_dic, template_path)

    with pipeline_step("Online steps"):
        reference_query = obj_templates_average_feats[:, :, 0, :]

        for i, test_sample in tqdm(
            enumerate(test_dl), desc="Testing", total=len(test_ds)
        ):
            test_image = test_sample["image"].to(device)

            H_org, W_org = test_image.shape[-2:]

            test_embedding = descriptor.encode_image(test_image)[0]
            test_embedding_key = test_embedding[1, :, :, :]
            C, _, _ = test_embedding_key.shape

            reference_query_mean = reference_query.mean(dim=1)
            test_embedding_key_flatten = test_embedding_key.flatten(1, 2)

            reference_query_mean /= torch.norm(
                reference_query_mean, dim=-1, keepdim=True
            )
            test_embedding_key_flatten /= torch.norm(
                test_embedding_key_flatten, dim=-1, keepdim=True
            )

            Q_ref_key_scene_sim = reference_query_mean @ test_embedding_key_flatten

            Q_ref_key_scene_sim = (Q_ref_key_scene_sim - Q_ref_key_scene_sim.min()) / (
                Q_ref_key_scene_sim.max() - Q_ref_key_scene_sim.min()
            )

            foreground_prompt_map = Q_ref_key_scene_sim.view(
                Q_ref_key_scene_sim.shape[0], *descriptor.output_spatial_size
            )
            foreground_prompt_map = foreground_prompt_map > 0.8
            foreground_prompt_map = foreground_prompt_map[0].bool()

            with pipeline_step("Visualizing the results", speak=False):
                result = descriptor.inv_trans(test_image)

                foreground_prompt_map = (
                    foreground_prompt_map.unsqueeze(-1).permute(2, 0, 1).float()
                )
                foreground_prompt_map = descriptor.to_org_size(
                    foreground_prompt_map, H_org, W_org
                )
                foreground_prompt_map[foreground_prompt_map == 0] = 0.2
                result *= foreground_prompt_map
                result = descriptor.cropper.reverse(result, test_sample["org_size"][0])
                vutils.save_image(
                    result, f"temp/main/{descriptor.name}_{i}_positives.png"
                )

            if i == 10:
                break

        # miou_list = np.array(miou_list)
        # # sort it
        # miou_list = miou_list[miou_list[:, 2].argsort()]
        # np.savetxt("temp/main/miou_list.txt", miou_list, fmt="%f")
        # mean_iou = miou_list[:, 2].mean()
        # print(f"Mean iou: {mean_iou}")


if __name__ == "__main__":
    seed_all(41)
    run_pipeline(device)
