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

from dataset.bop import BOP_dataset, BOP_dataset_info, BOP_obj_template_dataset
from descriptor.dino_wraper import Dino_Descriptors
from descriptor.sam_wraper import Sam_Wraper
from pipeline_util import (
    pipeline_step,
    seed_all,
    extract_features,
    new_classification,
    convert_to_coco_format,
)
from segmentation import (
    SAM_Seg,
    SAM2_Seg,
)
from segmentation.util import (
    get_bounding_boxes_batch,
    mask_miou_batch,
    remove_very_small_detections,
)
from segmentation.prompt_helper import (
    sim_2_point_prompts_2,
    sim_prompts_filter,
    sim_prompts_filter_2,
    generate_patch_grid_points,
    generate_grid_points,
)

from segmentation.automatic_mask_generator import SamAutomaticMaskGenerator
from segmentation.sam6d import CustomSamAutomaticMaskGenerator


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
    "num_templates_per_obj": 42,
}
sam_settings = {"class": SAM_Seg, "model_type": "vit_h"}
sam2_settings = {"class": SAM2_Seg, "model_type": "hiera_l"}


descriptor_setting = (
    descriptor_settings_dino_descriptor  # descriptor_settings_dino_descriptor
)
detector_setting = sam_settings

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
        detector = detector_setting["class"](detector_setting["model_type"], device)

        template_path = (
            f"offline_data/lab_sam6d/{descriptor.name}/{dataset_name}/templates.pth"
        )
        rgb_path = os.path.join(os.path.dirname(template_path), "rgb")
        mask_path = os.path.join(os.path.dirname(template_path), "mask")
        pipeline_step.print_full_width(
            f"loading from the test split of {dataset_name} dataset"
        )

        test_ds = BOP_dataset(
            **dataset_settings,
            **test_dataset_settings,
            transform_image=descriptor.transform,
            # cropper=descriptor.cropper,
            enhance_rgb=True,
        )
        test_ds.prepare_metadata()
        test_dl = test_ds.get_dataloader(batch_size=1, num_workers=4, shuffle=True)

    with pipeline_step("Offline preparation"):
        if os.path.exists(template_path):
            pipeline_step.print_full_width("Loading the object's templates features")
            template_dic = torch.load(template_path)

            obj_templates_feats_scaled = template_dic["obj_templates_feats_scaled"]
            obj_templates_average_feats_scaled = template_dic[
                "obj_templates_average_feats_scaled"
            ]
            obj_templates_cls_scaled = template_dic[
                "obj_templates_average_cls_scaled"
            ]
            obj_average_feats_all_layers_scaled = template_dic[
                "obj_average_feats_all_layers_scaled"
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

                obj_templates_feats_scaled = []
                obj_templates_average_feats_scaled = []
                obj_templates_cls_scaled = []
                obj_average_feats_all_layers_scaled = []
                for i, sample in enumerate(
                    tqdm(obj_template_dl, desc="Adding features to the point cloud")
                ):
                    image = sample["image"][0].to(device)
                    masks = sample["mask"][0].to(device)
                    image *= masks

                    obj_average_feats_scaled = None
                    for iter in range(1 + 0):
                        (
                            _,
                            _,
                            scaled_obj_templates_feat,
                            scaled_obj_templates_average_feat,
                            scaled_obj_average_feats_all_layers,
                            _,
                            _,
                        ) = extract_features(
                            image,
                            masks,
                            descriptor,
                            inplane_rotation=False,
                            batch_size=1,
                            g_info=obj_average_feats_scaled,
                        )
                        obj_average_feats_scaled = scaled_obj_average_feats_all_layers

                    obj_templates_feats_scaled.append(scaled_obj_templates_feat)
                    obj_templates_average_feats_scaled.append(
                        scaled_obj_templates_average_feat[:,0,:]
                    )
                    obj_templates_cls_scaled.append(scaled_obj_templates_average_feat[:,1,:])
                    obj_average_feats_all_layers_scaled.append(obj_average_feats_scaled)

                obj_templates_feats_scaled = torch.stack(obj_templates_feats_scaled)
                obj_templates_average_feats_scaled = torch.stack(
                    obj_templates_average_feats_scaled
                )
                obj_templates_cls_scaled = torch.stack(
                    obj_templates_cls_scaled
                )
                obj_average_feats_all_layers_scaled = torch.stack(
                    obj_average_feats_all_layers_scaled
                )
                sample = None

                # saving the object's templates features
                pipeline_step.print_full_width("Saving the object's templates features")

                template_dic = {}
                template_dic["obj_templates_feats_scaled"] = obj_templates_feats_scaled
                template_dic["obj_templates_average_feats_scaled"] = (
                    obj_templates_average_feats_scaled
                )
                template_dic["obj_templates_average_cls_scaled"] = (
                    obj_templates_cls_scaled
                )
                template_dic["obj_average_feats_all_layers_scaled"] = (
                    obj_average_feats_all_layers_scaled
                )

                torch.save(template_dic, template_path)

    with pipeline_step("Online steps"):
        all_results_list = []
        for i, test_sample in tqdm(
            enumerate(test_dl), desc="Testing", total=len(test_ds)
        ):
            scene_id = int(test_sample["scene_im_info"][0].scene_id)

            start_time = time()
            test_image = test_sample["image"].to(device)
            H_org, W_org = test_image.shape[-2:]

            test_embedding = descriptor.encode_image(test_image)[0][
                -1
            ]  # last layer
            # test_embedding_cls = test_embedding[4,:,0,0] #cls token is repeated in all patches
            test_embedding = test_embedding[3]
            C, _, _ = test_embedding.shape
            test_embedding = test_embedding.view(C, -1).permute(1, 0)
            test_embedding /= torch.norm(test_embedding, dim=-1, keepdim=True)

            with pipeline_step("Generating segmentation masks", speak=False):
                with pipeline_step("Generating prompts", speak=False):
                    # grid_prompt_locations = generate_patch_grid_points(
                    #     descriptor.output_spatial_size,
                    #     descriptor.patch_size,
                    #     device=device,
                    #     corners=True,
                    # )
                    # foreground_prompt_map, foreground_prompt_locations = (
                    #     sim_2_point_prompts(
                    #         scene_obj_sim=scene_obj_sim,
                    #         grid_prompt_locations=grid_prompt_locations,
                    #         spatial_size=descriptor.output_spatial_size,
                    #         threshold=0.5,
                    #     )
                    # )
                    # foreground_prompt_locations = foreground_prompt_locations.flatten(0,1)

                    grid_prompt_locations = generate_grid_points(
                        point_size=(32, 32),
                        target_size=test_image.shape[-2:],
                        device=device,
                    )
                    foreground_prompt_locations = grid_prompt_locations.view(-1, 2)

                with pipeline_step("Segmenting by prompts", speak=False):
                    test_image_sized, point_prompt_scaled, _, _ = (
                        detector.scale_image_prompt_to_dim(
                            image=descriptor.inv_trans(test_image),
                            point_prompt=foreground_prompt_locations,
                            # max_size=detector.input_size,
                        )
                    )
                    _ = detector.encode_image(
                        test_image_sized,
                        original_image_size=test_image.shape[-2:],
                    )

                    seg_masks, seg_scores = detector.segment_by_prompt(
                        prompt=point_prompt_scaled,
                        batch_size=64,
                        score_threshould=0.88,
                        stability_thresh=0.85,
                    )
                    seg_masks = seg_masks > 0

                with pipeline_step("Post processing", speak=False):
                    seg_boxes = get_bounding_boxes_batch(seg_masks)
                    keep_idxs = batched_nms(
                        boxes=seg_boxes.float(),
                        scores=seg_scores,
                        idxs=torch.zeros(len(seg_masks)).to(device),  # categories
                        iou_threshold=0.7,
                    )
                    seg_masks = seg_masks[keep_idxs]
                    seg_scores = seg_scores[keep_idxs]
                    seg_boxes = seg_boxes[keep_idxs]

                    keep_idxs = remove_very_small_detections(seg_masks, boxes=seg_boxes)
                    seg_masks = seg_masks[keep_idxs]
                    seg_scores = seg_scores[keep_idxs]
                    seg_boxes = seg_boxes[keep_idxs]

                    # tvutils.save_image(
                    #     seg_masks.float().unsqueeze(1),
                    #     f"temp/seg_masks_{i}.png",
                    # )

            with pipeline_step("Segmentation classification and scoring", speak=False):
                bbox = get_bounding_boxes_batch(seg_masks.squeeze(1))
                test_image_cropped_scaled = descriptor.scaled_cropper(
                    test_image.repeat(len(bbox), 1, 1, 1), bbox
                )
                seg_masks_cropped_scaled = descriptor.scaled_cropper(
                    seg_masks.unsqueeze(1).float(), bbox
                )

                (
                    test_image_cropped_scaled_embedding,
                    test_image_cropped_scaled_average_embedding,
                    seg_masks_cropped_scaled_patched,
                ) = descriptor.encode_image(
                    image=test_image_cropped_scaled,
                    mask=seg_masks_cropped_scaled,
                    is_scaled=True,
                )
                # last layer and token (3)
                test_image_cropped_scaled_embedding = (
                    test_image_cropped_scaled_embedding[:, -1, 3]
                )
                test_image_cropped_scaled_cls = (
                    test_image_cropped_scaled_average_embedding[:, -1, 4]
                )
                test_image_cropped_scaled_average_embedding = (
                    test_image_cropped_scaled_average_embedding[:, -1, 3]
                )

                with pipeline_step("Classification scoring", speak=False):
                    (
                        estimated_obj_idx,
                        best_selected_template_idx,
                        estimated_score_cls,
                    ) = new_classification(
                        test_image_cropped_scaled_cls,
                        obj_templates_cls_scaled,
                        k_view=5,
                    )

                    # filtering the results
                    cls_score_filter = estimated_score_cls > 0.2
                    estimated_obj_idx = estimated_obj_idx[cls_score_filter]
                    best_selected_template_idx = best_selected_template_idx[
                        cls_score_filter
                    ]
                    estimated_score_cls = estimated_score_cls[cls_score_filter]
                    seg_masks = seg_masks[cls_score_filter]
                    seg_scores = seg_scores[cls_score_filter]
                    test_image_cropped_scaled_embedding = (
                        test_image_cropped_scaled_embedding[cls_score_filter]
                    )
                    estimated_obj_ids = torch.tensor(
                        BOP_dataset_info[dataset_name]["obj_ids"], device=device
                    )[estimated_obj_idx]

                with pipeline_step("Appearance scoring", speak=False):
                    selected_obj_template_feats = obj_templates_feats_scaled[
                        estimated_obj_idx, best_selected_template_idx
                    ].permute(0, 2, 3, 1)
                    selected_obj_template_feats = selected_obj_template_feats.flatten(
                        1, 2
                    )
                    appearance_embedding = test_image_cropped_scaled_embedding.permute(
                        0, 2, 3, 1
                    ).flatten(1, 2)

                    # selected_obj_template_feats_mask = (
                    #     selected_obj_template_feats.sum(dim=-1) > 0
                    # )
                    # appearance_embedding_mask = appearance_embedding.sum(dim=-1) > 0
                    # appearance_sim_mask = appearance_embedding_mask.unsqueeze(
                    #     2
                    # ) * selected_obj_template_feats_mask.unsqueeze(1)

                    app_b, app_p, _ = appearance_embedding.shape

                    appearance_embedding /= torch.norm(
                        appearance_embedding, dim=-1, keepdim=True
                    )
                    selected_obj_template_feats /= torch.norm(
                        selected_obj_template_feats, dim=-1, keepdim=True
                    )

                    appearance_sim = torch.bmm(
                        appearance_embedding,
                        selected_obj_template_feats.transpose(1, 2),
                    )
                    # appearance_sim = appearance_sim * appearance_sim_mask
                    estimated_score_appearance = appearance_sim.max(dim=1).values
                    estimated_score_appearance = estimated_score_appearance.sum(
                        dim=-1
                    ) / ((estimated_score_appearance > 0).sum(dim=-1) + 1e-6)
                    estimated_score_appearance = estimated_score_appearance.clamp(0, 1)

                end_time = time()
                elapassed_time = end_time - start_time
                # seg_masks_org = descriptor.cropper.reverse(
                #     seg_masks.unsqueeze(0).float(), test_sample["org_size"][0]
                # )[0].squeeze(0)

                coco_bbox = get_bounding_boxes_batch(seg_masks)
                coco_bbox[:, 2] = coco_bbox[:, 2] - coco_bbox[:, 0]
                coco_bbox[:, 3] = coco_bbox[:, 3] - coco_bbox[:, 1]

                seg_masks_gt = (
                    test_sample["scene_im_info"][0]
                    .obj_gt.visible_masks.squeeze(1)
                    .to(device)
                )
                seg_id_gt = test_sample["scene_im_info"][0].obj_gt.obj_id.to(device)
                # find the maximum and not integrated miou for each mask from seg_masks over seg_masks_gt
                all_mious = mask_miou_batch(seg_masks.bool(), seg_masks_gt)
                gt_score, all_mious_idx = all_mious.max(dim=1)

                seg_scores = estimated_score_cls  # (estimated_score_appearance + estimated_score_cls) / 2

                # gt id and estimated score
                # estimated_obj_ids = seg_id_gt[all_mious_idx]

                # for j in range(len(all_mious_idx)):
                #     if gt_score[j] >= 0.4:
                #         if seg_id_gt[all_mious_idx[j]] != estimated_obj_ids[j]:
                #             gt_score[j] *= 0.5
                #     else:
                #         gt_score[j] = 0.01
                # # estimated id and gt score
                # seg_scores = gt_score

                keep_idxs = batched_nms(
                    get_bounding_boxes_batch(seg_masks).float(),
                    seg_scores,
                    estimated_obj_ids,
                    0.25,
                )
                seg_masks = seg_masks[keep_idxs]
                seg_scores = seg_scores[keep_idxs]
                estimated_obj_ids = estimated_obj_ids[keep_idxs]
                coco_bbox = coco_bbox[keep_idxs]

                if len(seg_masks) == 0:
                    continue
                current_result = convert_to_coco_format(
                    scene_id=int(test_sample["scene_im_info"][0].scene_id),
                    image_id=int(test_sample["scene_im_info"][0].im_id),
                    estimations=seg_masks.cpu(),
                    scores=seg_scores.cpu(),  # topk_obj_score.squeeze(1).cpu(),
                    obj_ids=estimated_obj_ids.cpu(),
                    bboxes=coco_bbox.cpu(),
                    time=elapassed_time / len(seg_masks),
                )
                for res in current_result:
                    all_results_list.append(res)

                # if i == 30:
                #     break

        result_filename = "results/lab_sam6d/est-grid32-clstoken_tless-test.json"
        with open(result_filename, "w") as f:
            json.dump(all_results_list, f)


if __name__ == "__main__":
    seed_all(41)
    run_pipeline(device)


def compute_semantic_score(proposal_decriptors, obj_descriptors):
    # compute matching scores for each proposals
    scores = cosine_sim(
        proposal_decriptors, obj_descriptors
    )  # N_proposals x N_objects x N_templates

    score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
    score_per_proposal_and_object = torch.mean(score_per_proposal_and_object, dim=-1)

    # assign each proposal to the object with highest scores
    score_per_proposal, assigned_idx_object = torch.max(
        score_per_proposal_and_object, dim=-1
    )  # N_query

    idx_selected_proposals = torch.arange(
        len(score_per_proposal), device=score_per_proposal.device
    )[score_per_proposal > 0.2]
    pred_idx_objects = assigned_idx_object[idx_selected_proposals]
    semantic_score = score_per_proposal[idx_selected_proposals]

    # compute the best view of template
    flitered_scores = scores[idx_selected_proposals, ...]
    best_template = best_template_pose(flitered_scores, pred_idx_objects)

    return idx_selected_proposals, pred_idx_objects, semantic_score, best_template


from torch.nn import functional as F


def cosine_sim(query, reference):
    N_query = query.shape[0]
    N_objects, N_templates = reference.shape[0], reference.shape[1]
    references = reference.clone().unsqueeze(0).repeat(N_query, 1, 1, 1)
    queries = query.clone().unsqueeze(1).repeat(1, N_templates, 1)
    queries = F.normalize(queries, dim=-1)
    references = F.normalize(references, dim=-1)

    similarity = BatchedData(batch_size=None)
    for idx_obj in range(N_objects):
        sim = F.cosine_similarity(
            queries, references[:, idx_obj], dim=-1
        )  # N_query x N_templates
        similarity.append(sim)
    similarity.stack()
    similarity = similarity.data
    similarity = similarity.permute(1, 0, 2)  # N_query x N_objects x N_templates
    return similarity.clamp(min=0.0, max=1.0)


def best_template_pose(scores, pred_idx_objects):
    _, best_template_idxes = torch.max(scores, dim=-1)
    N_query, N_object = best_template_idxes.shape[0], best_template_idxes.shape[1]
    pred_idx_objects = pred_idx_objects[:, None].repeat(1, N_object)

    assert N_query == pred_idx_objects.shape[0], "Prediction num != Query num"

    best_template_idx = torch.gather(
        best_template_idxes, dim=1, index=pred_idx_objects
    )[:, 0]

    return best_template_idx


class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)
