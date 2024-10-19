import os
from time import time

# from segmentation.sam2 import SAM2_Wraper
import torch
import torchvision
import torch.utils
from torchvision import utils as tvutils
from torchvision.ops import batched_nms
from tqdm import tqdm
import numpy as np
import json
from torchvision import utils as vutils
from PIL import Image

from dataset.bop import BOP_dataset, BOP_dataset_info, BOP_obj_template_dataset
from descriptor.dino_wraper import Dino_Descriptors, Dino_Wraper
from descriptor.sam_wraper import Sam_Wraper

from pipeline_util import pipeline_step, seed_all, extract_features
import matplotlib.pyplot as plt


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
        template_path = f"offline_data/{descriptor.name}/{dataset_name}/templates.pth"
        rgb_path = os.path.join(os.path.dirname(template_path), "rgb")
        mask_path = os.path.join(os.path.dirname(template_path), "mask")

    with pipeline_step("Offline preparation"):
        if os.path.exists(template_path):
            pipeline_step.print_full_width("Loading the object's templates features")
            template_dic = torch.load(template_path)
            # # obj_templates_feats = template_dic["obj_templates_feats"]
            # obj_templates_average_feats = template_dic["obj_templates_average_feats"]

            # obj_templates_feats_scaled = template_dic["obj_templates_feats_scaled"]
            # obj_templates_average_feats_scaled = template_dic[
            #     "obj_templates_average_feats_scaled"
            # ]

            objs_average_feats_scaled = template_dic["objs_average_feats_scaled"]
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
                    visible_fract_threshold=0.9,
                    num_templates=descriptor.num_templates_per_obj,
                    # cropper=descriptor.cropper,  # CropResizePad(224)
                )
                obj_template_dl = obj_template_ds.get_dataloader(
                    batch_size=1, num_workers=2, shuffle=False
                )

                os.makedirs(os.path.dirname(template_path), exist_ok=True)
                os.makedirs(rgb_path, exist_ok=True)
                os.makedirs(mask_path, exist_ok=True)

                obj_templates_feats = []
                obj_templates_average_feats = []
                obj_templates_feats_scaled = []
                obj_templates_average_feats_scaled = []
                objs_average_feats_scaled = []
                for i, sample in enumerate(
                    tqdm(obj_template_dl, desc="Adding features to the point cloud")
                ):
                    image = sample["image"][0].to(device)
                    masks = sample["mask"][0].to(device)
                    image *= masks

                    obj_average_feats_scaled = None
                    for iter in range(1 + 0):
                        (
                            obj_templates_feat,
                            obj_templates_average_feat,
                            scaled_obj_templates_feat,
                            scaled_obj_templates_average_feat,
                            scaled_images,
                            scaled_masks,
                        ) = extract_features(
                            image,
                            masks,
                            descriptor,
                            inplane_rotation=False,
                            batch_size=1,
                            g_info=obj_average_feats_scaled,
                        )
                        obj_average_feats_scaled = (
                            scaled_obj_templates_average_feat.mean(dim=0)
                        )

                    for j in range(len(image)):
                        tvutils.save_image(
                            descriptor.inv_trans(scaled_images[j]),
                            os.path.join(
                                rgb_path,
                                f"{i}_{j}.png",
                            ),
                        )
                        tvutils.save_image(
                            scaled_masks[j],
                            os.path.join(
                                mask_path,
                                f"{i}_{j}.png",
                            ),
                        )

                    # obj_templates_feats.append(obj_templates_feat)
                    # obj_templates_average_feats.append(obj_templates_average_feat)
                    # obj_templates_feats_scaled.append(scaled_obj_templates_feat)
                    # obj_templates_average_feats_scaled.append(
                    #     scaled_obj_templates_average_feat
                    # )
                    objs_average_feats_scaled.append(obj_average_feats_scaled)
                # obj_templates_feats = torch.stack(obj_templates_feats)
                # obj_templates_average_feats = torch.stack(obj_templates_average_feats)

                # obj_templates_feats_scaled = torch.stack(obj_templates_feats_scaled)
                # obj_templates_average_feats_scaled = torch.stack(
                #     obj_templates_average_feats_scaled
                # )
                objs_average_feats_scaled = torch.stack(objs_average_feats_scaled)
                sample = None

                # saving the object's templates features
                pipeline_step.print_full_width("Saving the object's templates features")

                template_dic = {}
                # template_dic["obj_templates_feats"] = obj_templates_feats
                # template_dic["obj_templates_average_feats"] = (
                #     obj_templates_average_feats
                # )
                # template_dic["obj_templates_feats_scaled"] = obj_templates_feats_scaled
                # template_dic["obj_templates_average_feats_scaled"] = (
                #     obj_templates_average_feats_scaled
                # )
                template_dic["objs_average_feats_scaled"] = objs_average_feats_scaled

                torch.save(template_dic, template_path)

    with pipeline_step("Online steps"):
        obj_id = 0
        obj_template_id = 6

        reference_token = objs_average_feats_scaled[obj_id, -1, 3, :]
        reference_token = reference_token.unsqueeze(0)

        for i, test_sample in tqdm(
            enumerate(test_dl), desc="Testing", total=len(test_ds)
        ):
            test_image = test_sample["image"].to(device)

            H_org, W_org = test_image.shape[-2:]

            test_embedding = descriptor.encode_image(
                test_image, g_info=objs_average_feats_scaled[0, :, :, :]
            )[0][-1]
            test_embedding_token = (
                test_embedding[3, :, :, :].flatten(1, 2).permute(1, 0)
            )
            for head in range(17):
                head_range_start = head * 64
                head_range_end = (head + 1) * 64
                if head < 16:
                    reference_query_head = reference_token[
                        :, head_range_start:head_range_end
                    ]
                    test_embedding_query_head = test_embedding_token[
                        :, head_range_start:head_range_end
                    ]

                else:
                    reference_query_head = reference_token
                    test_embedding_query_head = test_embedding_token

                reference_query_head /= reference_query_head.norm(dim=-1, keepdim=True)
                test_embedding_query_head /= test_embedding_query_head.norm(
                    dim=-1, keepdim=True
                )
                fore_sim = test_embedding_query_head @ reference_query_head.t()

                fore_sim = fore_sim.view(-1, *descriptor.output_spatial_size)
                fore_sim = torch.nn.functional.interpolate(
                    fore_sim.unsqueeze(0), size=(H_org, W_org)
                )
                
                print(
                    f"sim_max: {fore_sim.max()} and sim_min: {fore_sim.min()}, difference: {fore_sim.max() - fore_sim.min()}"
                )
                # fore_sim[fore_sim < 0.4] = 0
                fore_sim = fore_sim.squeeze(0).squeeze(0).cpu().numpy()

                plt.figure(figsize=(10, 10))

                plt.imshow(
                    descriptor.inv_trans(test_image)
                    .cpu()
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .numpy()
                )
                plt.imshow(
                    fore_sim, cmap="jet", alpha=0.4
                )  # Overlay attention map with transparency
                plt.title("Attention weights")

                plt.savefig(
                    f"temp/lab_attention_scene/token_token/attention_weights{i}_{obj_id}_{head}.png"
                )
                plt.close()

            break


if __name__ == "__main__":
    seed_all(41)
    run_pipeline(device)
