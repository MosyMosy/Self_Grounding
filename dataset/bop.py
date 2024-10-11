import json
import os
import imageio.v3 as iio
from PIL import Image, ImageEnhance
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix
from pytorch3d.ops import sample_farthest_points
from operator import itemgetter
from torchvision.utils import make_grid, save_image
from dataset.util import CropResizePad_v2, depth_image_to_pointcloud
from time import time
from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes

import matplotlib.pyplot as plt

# from dataset.util import equalize_brightness, constant_brightness


BOP_dataset_info = {
    "LM-O": {
        "name": "LM-O",
        "obj_ids": [1, 5, 6, 8, 9, 10, 11, 12],
        "train_dir": "train_pbr",
        "test_dir": "test",
        "val_dir": "",
        "image_name_template": "rgb/{0}.jpg",
        "depth_name_template": "depth/{0}.png",
    },
    "tless": {
        "name": "T-LESS",
        "obj_ids": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        ],
        "obj_groups": [
            1,
            1,
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            8,
            8,
            9,
            9,
            10,
            11,
            12,
            12,
            13,
            13,
            13,
            14,
        ],
        "train_dir": "train_pbr",
        "test_dir": "test_primesense",
        "val_dir": "",
        "image_name_template": "rgb/{0}.jpg",
        "depth_name_template": "depth/{0}.png",
    },
    "ITODD": {
        "name": "ITODD",
        "obj_ids": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
        ],
        "train_dir": "train_pbr",
        "test_dir": "test",
        "val_dir": "val",
        "image_name_template": "rgb/{0}.jpg",
        "depth_name_template": "depth/{0}.png",
    },
    "HB": {
        "name": "HB",
        "obj_ids": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
        ],
        "train_dir": "train_pbr",
        "test_dir": "test_primesense",
        "val_dir": "",
        "image_name_template": "rgb/{0}.jpg",
        "depth_name_template": "depth/{0}.png",
    },
    "YCB-V": {
        "name": "YCB-V",
        "obj_ids": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
        ],
        "train_dir": "train_pbr",
        "test_dir": "test",
        "val_dir": "",
        "image_name_template": "rgb/{0}.jpg",
        "depth_name_template": "depth/{0}.png",
    },
    "IC-BIN": {
        "name": "IC-BIN",
        "obj_ids": [1, 2],
        "train_dir": "train_pbr",
        "test_dir": "test",
        "val_dir": "",
        "image_name_template": "rgb/{0}.jpg",
        "depth_name_template": "depth/{0}.png",
    },
    "TUD-L": {
        "name": "TUD-L",
        "obj_ids": [1, 2, 3],
        "train_dir": "train_pbr",
        "test_dir": "test",
        "val_dir": "",
        "image_name_template": "rgb/{0}.jpg",
        "depth_name_template": "depth/{0}.png",
    },
}


class BOP_dataset:
    def __init__(
        self,
        root_dir,
        split,
        mode,
        image_name_template="rgb/{0}.png",
        depth_name_template="depth/{0}.png",
        transform_image=None,
        get_obj_gt_visible_masks=False,
        dataset_info=None,
        cropper=None,
        enhance_rgb=False,
    ) -> None:
        assert mode in ["train", "test"], f"mode {mode} not supported"
        assert os.path.isdir(root_dir), f"root_dir {root_dir} does not exist"
        assert os.path.isdir(
            os.path.join(root_dir, split)
        ), f"split {split} does not exist in {root_dir}"
        self.root_dir = root_dir
        self.split = split
        self.split_path = os.path.join(root_dir, split)
        self.mode = mode
        self.image_name_template = image_name_template
        self.depth_name_template = depth_name_template

        if transform_image is None:
            self.transform_image = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform_image = transform_image

        self.get_obj_gt_visible_masks = get_obj_gt_visible_masks
        self.dataset_info = dataset_info

        # metadata dictionaries
        self.metadata_dic = {}
        self.cropper = cropper
        self.enhance_rgb = enhance_rgb

    def prepare_metadata(self):
        test_targets_dic = {}
        if self.mode == "test":
            test_targets_path = os.path.join(self.root_dir, "test_targets_bop19.json")
            with open(test_targets_path, "r") as file:
                test_targets = json.load(file)

            for item in test_targets:
                name = (
                    str(item["scene_id"]).zfill(6) + "_" + str(item["im_id"]).zfill(6)
                )
                if name in test_targets_dic:
                    test_targets_dic[name]["obj_ids"].append(item["obj_id"])
                    test_targets_dic[name]["inst_counts"].append(item["inst_count"])
                else:
                    test_targets_dic[name] = {
                        "obj_ids": [item["obj_id"]],
                        "inst_counts": [item["inst_count"]],
                    }
        all_directories = list(os.scandir(self.split_path))
        for scene_folder in tqdm(
            all_directories,
            total=len(all_directories),
            desc=f"Preparing dataset's metadata",
        ):
            if (
                scene_folder.is_dir()
                and scene_folder.name.isdigit()
                and len(scene_folder.name) == 6
            ):
                scene_camera_path = os.path.join(
                    self.split_path, scene_folder, "scene_camera.json"
                )
                scene_gt_path = os.path.join(
                    self.split_path, scene_folder, "scene_gt.json"
                )
                scene_gt_info_path = os.path.join(
                    self.split_path, scene_folder, "scene_gt_info.json"
                )
                # reading the camera, scene_gt and scene_gt_info files
                # scene_camera: Overall information about the scene and the camera
                # scene_gt: object ground truth information like pose
                # scene_gt_info: object ground truth information like bb and visibility
                with open(scene_camera_path, "r") as scene_camera_file, open(
                    scene_gt_path, "r"
                ) as scene_gt_file, open(scene_gt_info_path, "r") as scene_gt_info_file:
                    scene_camera = json.load(scene_camera_file)
                    scene_gt = json.load(scene_gt_file)
                    scene_gt_info = json.load(scene_gt_info_file)
                for key, value in scene_camera.items():
                    im_id = str(key).zfill(6)
                    name = scene_folder.name + "_" + im_id
                    scene_im_dic = {}
                    scene_im_dic["scene_id"] = scene_folder.name
                    scene_im_dic["im_id"] = im_id
                    scene_im_dic.update(value)
                    # adding scene_gt to the metadata dictionary
                    scene_im_dic["obj_gt"] = scene_gt[key]
                    # adding scene_gt_info to the scene_gt of the metadata dictionary
                    for i in range(len(scene_im_dic["obj_gt"])):
                        scene_im_dic["obj_gt"][i].update(scene_gt_info[key][i])

                    if self.mode == "test":
                        scene_im_dic["obj_ids"] = test_targets_dic[name]["obj_ids"]
                        scene_im_dic["inst_counts"] = test_targets_dic[name][
                            "inst_counts"
                        ]

                    self.metadata_dic[name] = scene_im_dic

    def __len__(self):
        if self.metadata_dic == {}:
            self.prepare_metadata()
        return len(self.metadata_dic)

    def __getitem__(self, *args):
        key = ""
        if len(args) == 1 and isinstance(args[0], str):
            key = args[0]
        elif len(args) == 2 and all(isinstance(arg, int) for arg in args):
            key = str(args[0]).zfill(6) + "_" + str(args[1]).zfill(6)
        elif len(args) == 1 and isinstance(args[0], int):
            key = key = list(self.metadata_dic)[args[0]]
        else:
            raise ValueError("Invalid arguments")

        scene_path = os.path.join(self.split_path, self.metadata_dic[key]["scene_id"])
        im_id = self.metadata_dic[key]["im_id"]

        image_path = os.path.join(
            scene_path,
            self.image_name_template.format(im_id),
        )
        depth_path = os.path.join(
            scene_path,
            self.depth_name_template.format(im_id),
        )
        image = Image.open(image_path)
        if self.enhance_rgb:
            contrast = ImageEnhance.Contrast(image)
            color = ImageEnhance.Color(image)
            sharpness = ImageEnhance.Sharpness(image)
            image = color.enhance(1.8)
            image = contrast.enhance(2.5)
            image = sharpness.enhance(2)
        if image.mode == "RGB":
            image = self.transform_image(image)
        org_size = (image.shape[-2], image.shape[-1])

        depth = torch.from_numpy(np.array(iio.imread(depth_path))).to(torch.float32)

        xyz = depth_image_to_pointcloud(
            (depth.unsqueeze(0)),
            scale=torch.tensor(self.metadata_dic[key]["depth_scale"]).unsqueeze(0),
            K=torch.tensor(self.metadata_dic[key]["cam_K"]).view(1, 3, 3),
        )[0].permute(2, 0, 1)

        obj_gt_visible_masks = []
        if self.get_obj_gt_visible_masks:
            for i in range(len(self.metadata_dic[key]["obj_gt"])):
                mask_path = os.path.join(
                    scene_path,
                    "mask_visib",
                    im_id + "_" + str(i).zfill(6) + ".png",
                )
                mask = Image.open(mask_path)
                obj_gt_visible_masks.append(transforms.ToTensor()(mask))

        depth = depth.unsqueeze(-1).permute(2, 0, 1)
        if self.cropper is not None:
            image = self.cropper(image.unsqueeze(0))[0]
            depth = self.cropper(depth.unsqueeze(0))[0]
            xyz = self.cropper(xyz.unsqueeze(0))[0]

        depth_rgb = depth.squeeze(0).numpy()
        depth_rgb = (depth_rgb - depth_rgb.min()) / (depth_rgb.max() - depth_rgb.min())
        colormap = plt.get_cmap("plasma")
        depth_rgb = colormap(depth_rgb)[:, :, :3]
        depth_rgb = torch.from_numpy(depth_rgb).permute(2, 0, 1)

        return {
            "image": image,
            "depth": depth,
            "depth_rgb": depth_rgb,
            "xyz": xyz,
            "metadata": self.metadata_dic[key],
            "obj_gt_visible_masks": obj_gt_visible_masks,
            "org_size": org_size,
        }

    def get_dataloader(self, batch_size, shuffle, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=BOP_dataset.BOP_dataset_collate_fn,
        )

    @staticmethod
    def BOP_dataset_collate_fn(batch):
        images = torch.stack([data["image"] for data in batch])
        depths = torch.stack([data["depth"] for data in batch])
        depth_rgb = torch.stack([data["depth_rgb"] for data in batch])
        xyzs = torch.stack([data["xyz"] for data in batch])
        org_sizes = [data["org_size"] for data in batch]

        scene_im_info = [
            BOP_dataset.BOP_scene_im_properties(
                **data["metadata"], obj_gt_visible_masks=data["obj_gt_visible_masks"]
            )
            for data in batch
        ]
        return {
            "image": images,
            "depth": depths,
            "depth_rgb": depth_rgb,
            "xyz": xyzs,
            "scene_im_info": scene_im_info,
            "org_size": org_sizes,
        }

    class BOP_obj_properties:
        def __init__(
            self,
            cam_R_m2c,
            cam_t_m2c,
            obj_id,
            bbox_obj,
            bbox_visib,
            px_count_all,
            px_count_valid,
            px_count_visib,
            visib_fract,
        ):
            # convert to tensor if it is not
            self.cam_R_m2c = (
                torch.tensor(cam_R_m2c)
                if not isinstance(cam_R_m2c, torch.Tensor)
                else cam_R_m2c
            )
            self.cam_t_m2c = (
                torch.tensor(cam_t_m2c)
                if not isinstance(cam_t_m2c, torch.Tensor)
                else cam_t_m2c
            )
            self.obj_id = (
                torch.tensor(obj_id) if not isinstance(obj_id, torch.Tensor) else obj_id
            )
            self.bbox_obj = (
                torch.tensor(bbox_obj)
                if not isinstance(bbox_obj, torch.Tensor)
                else bbox_obj
            )
            self.bbox_visib = (
                torch.tensor(bbox_visib)
                if not isinstance(bbox_visib, torch.Tensor)
                else bbox_visib
            )
            self.px_count_all = (
                torch.tensor(px_count_all)
                if not isinstance(px_count_all, torch.Tensor)
                else px_count_all
            )
            self.px_count_valid = (
                torch.tensor(px_count_valid)
                if not isinstance(px_count_valid, torch.Tensor)
                else px_count_valid
            )
            self.px_count_visib = (
                torch.tensor(px_count_visib)
                if not isinstance(px_count_visib, torch.Tensor)
                else px_count_visib
            )
            self.visib_fract = (
                torch.tensor(visib_fract)
                if not isinstance(visib_fract, torch.Tensor)
                else visib_fract
            )
            self.visible_masks = None

        def to_device(self, device):
            for attr, value in self.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(self, attr, value.to(device))
            return self

        @staticmethod
        def instance_list_to_one_instance(instance_list, visible_masks=None):
            cam_R_m2c = torch.stack([instance.cam_R_m2c for instance in instance_list])
            cam_t_m2c = torch.stack([instance.cam_t_m2c for instance in instance_list])
            obj_id = torch.stack([instance.obj_id for instance in instance_list])
            bbox_obj = torch.stack([instance.bbox_obj for instance in instance_list])
            bbox_visib = torch.stack(
                [instance.bbox_visib for instance in instance_list]
            )
            px_count_all = torch.stack(
                [instance.px_count_all for instance in instance_list]
            )
            px_count_valid = torch.stack(
                [instance.px_count_valid for instance in instance_list]
            )
            px_count_visib = torch.stack(
                [instance.px_count_visib for instance in instance_list]
            )
            visib_fract = torch.stack(
                [instance.visib_fract for instance in instance_list]
            )

            if visible_masks is not None:
                visible_masks_tensor = torch.stack(
                    [gt_mask for gt_mask in visible_masks]
                ).to(bool)
            else:
                visible_masks_tensor = None

            obj_properties = BOP_dataset.BOP_obj_properties(
                cam_R_m2c=cam_R_m2c.view(-1, 3, 3),
                cam_t_m2c=cam_t_m2c / 1000,
                obj_id=obj_id,
                bbox_obj=bbox_obj,
                bbox_visib=bbox_visib,
                px_count_all=px_count_all,
                px_count_valid=px_count_valid,
                px_count_visib=px_count_visib,
                visib_fract=visib_fract,
            )
            obj_properties.visible_masks = visible_masks_tensor
            return obj_properties

    class BOP_scene_im_properties:
        def __init__(
            self,
            scene_id,
            im_id,
            cam_K,
            depth_scale,
            obj_gt,
            obj_ids=[],
            inst_counts=[],
            mode=None,
            obj_gt_visible_masks=[],
            cam_R_w2c=None,
            cam_t_w2c=None,
            elev=None,
        ):
            self.scene_id = torch.tensor(int(scene_id))
            self.im_id = torch.tensor(int(im_id))
            self.cam_K = torch.tensor(cam_K).view(3, 3)
            self.depth_scale = torch.tensor(depth_scale)
            self.mode = mode
            self.obj_ids = torch.tensor(obj_ids)
            self.inst_counts = torch.tensor(inst_counts)

            self.obj_gt = [
                BOP_dataset.BOP_obj_properties(**obj_dic) for obj_dic in obj_gt
            ]
            self.obj_gt = BOP_dataset.BOP_obj_properties.instance_list_to_one_instance(
                self.obj_gt, obj_gt_visible_masks
            )

            self.cam_R_w2c = (
                None if cam_R_w2c is None else torch.tensor(cam_R_w2c).view(3, 3)
            )
            self.cam_t_w2c = (
                None if cam_t_w2c is None else torch.tensor(cam_t_w2c) / 1000
            )
            self.elev = None if elev is None else torch.tensor(elev)

        def to_device(self, device):
            for attr, value in self.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(self, attr, value.to(device))
                if attr == "obj_gt":
                    setattr(self, attr, [obj.to_device(device) for obj in value])
            return self

        # def get_visible_masks(self):
        #     if self.visible_masks is not None:
        #         return self.visible_masks
        #     else:


class BOP_obj_template_dataset(BOP_dataset):
    def __init__(
        self,
        root_dir,
        split,
        mode,
        image_name_template="rgb/{0}.png",
        depth_name_template="depth/{0}.png",
        transform_image=None,
        get_obj_gt_visible_masks=False,
        dataset_info=None,
        visible_fract_threshold=0.9,
        num_templates=42,
        cropper=None,
    ) -> None:
        if transform_image is None:
            transform_image = transforms.ToTensor()
        super().__init__(
            root_dir,
            split,
            mode,
            image_name_template,
            depth_name_template,
            transform_image,
            get_obj_gt_visible_masks,
            dataset_info,
        )

        self.visible_fract_threshold = visible_fract_threshold
        self.num_templates = num_templates
        self.cropper = cropper

        self.rotations, self.translations, _, self.scene_im_idxs = (
            self.sample_obj_rotations()
        )
        self.translations /= 1000

    def __len__(self):
        return len(self.dataset_info["obj_ids"])

    def sample_obj_rotations(self):
        if self.dataset_info is None:
            raise ValueError("dataset_info is not provided")
        if self.metadata_dic == {}:
            self.prepare_metadata()
        rotations = [[] for i in range(len(self.dataset_info["obj_ids"]))]
        translations = [[] for i in range(len(self.dataset_info["obj_ids"]))]
        visible_bboxs = [[] for i in range(len(self.dataset_info["obj_ids"]))]
        scene_im_idxs = [[] for i in range(len(self.dataset_info["obj_ids"]))]

        metadata_dic_values = list(self.metadata_dic.values())
        for item in tqdm(
            metadata_dic_values, desc="Getting all template bbox and rotations"
        ):
            scene_id = item["scene_id"]
            im_id = item["im_id"]
            for idx, obj_gt in enumerate(item["obj_gt"]):
                if obj_gt["visib_fract"] < self.visible_fract_threshold:
                    continue
                obj_indx = self.dataset_info["obj_ids"].index(obj_gt["obj_id"])
                rotations[obj_indx].append(obj_gt["cam_R_m2c"])
                translations[obj_indx].append(obj_gt["cam_t_m2c"])
                visible_bboxs[obj_indx].append(obj_gt["bbox_visib"])
                scene_im_idxs[obj_indx].append([scene_id, im_id, idx])

        selected_indexes = []
        for i in tqdm(range(len(rotations)), desc="Selecting farthest rotations"):
            rotations[i] = matrix_to_euler_angles(
                torch.tensor(rotations[i]).view(-1, 3, 3), convention="XYZ"
            )
            selected_indexes.append(
                sample_farthest_points(rotations[i].unsqueeze(0), K=self.num_templates)[
                    1
                ].squeeze(0)
            )

        rotations = [rotations[i][selected_indexes[i]] for i in range(len(rotations))]
        visible_bboxs = [
            itemgetter(*selected_indexes[i].tolist())(visible_bboxs[i])
            for i in range(len(visible_bboxs))
        ]
        scene_im_idxs = [
            itemgetter(*selected_indexes[i].tolist())(scene_im_idxs[i])
            for i in range(len(scene_im_idxs))
        ]
        translations = [
            itemgetter(*selected_indexes[i].tolist())(translations[i])
            for i in range(len(translations))
        ]

        rotations = torch.stack(rotations)
        rotations = euler_angles_to_matrix(rotations, convention="XYZ")
        visible_bboxs = torch.tensor(visible_bboxs)
        translations = torch.tensor(translations)

        return rotations, translations, visible_bboxs, scene_im_idxs

    def get_obj_pc(self, model_id, pc_size=1024):
        model_path = os.path.join(self.root_dir, "models")
        model_path = os.path.join(model_path, f"obj_{str(model_id).zfill(6)}.ply")
        verts, faces = load_ply(model_path)

        meshes = Meshes(verts=[verts], faces=[faces])
        model_points = sample_points_from_meshes(meshes, pc_size)
        return model_points / 1000

    def __getitem__(self, obj_idx):
        if self.dataset_info is None:
            raise ValueError("dataset_info is not provided")

        if self.metadata_dic == {}:
            self.prepare_metadata()

        rgb_images = []
        depth_images = []
        mask_images = []
        bboxes = []
        xyzs = []
        for item in self.scene_im_idxs[obj_idx]:
            rgb_path = os.path.join(
                self.split_path,
                str(item[0]).zfill(6),
                self.image_name_template.format(item[1]),
            )
            mask_path = os.path.join(
                self.split_path,
                str(item[0]).zfill(6),
                "mask_visib",
                item[1] + "_" + str(item[2]).zfill(6) + ".png",
            )
            depth_path = os.path.join(
                self.split_path,
                str(item[0]).zfill(6),
                self.depth_name_template.format(item[1]),
            )
            rgb = Image.open(rgb_path)

            depth = torch.from_numpy(np.array(iio.imread(depth_path))).to(torch.float32)

            mask = Image.open(mask_path)
            bbox = mask.getbbox()

            rgb = self.transform_image(rgb)
            mask = transforms.ToTensor()(mask)

            xyz = depth_image_to_pointcloud(
                (depth.unsqueeze(0) * mask),
                scale=torch.tensor(
                    self.metadata_dic[f"{item[0]}_{item[1]}"]["depth_scale"]
                ).unsqueeze(0),
                K=torch.tensor(self.metadata_dic[f"{item[0]}_{item[1]}"]["cam_K"]).view(
                    1, 3, 3
                ),
            )[0]
            rgb_images.append(rgb)  # (rgb * mask)
            depth = depth.unsqueeze(-1).permute(2, 0, 1)
            depth_images.append(depth * mask)
            mask_images.append(mask)
            bboxes.append(bbox)
            xyzs.append(xyz)
        rgb_images = torch.stack(rgb_images)
        depth_images = torch.stack(depth_images)
        mask_images = torch.stack(mask_images)
        bboxes = torch.tensor(bboxes)
        xyzs = torch.stack(xyzs).permute(0, 3, 1, 2)

        if self.cropper is not None:
            rgb_images = self.cropper(rgb_images, bboxes)
            depth_images = self.cropper(depth_images, bboxes)
            mask_images = self.cropper(mask_images, bboxes)
            xyzs = self.cropper(xyzs, bboxes)

        depth_rgb_batch = []
        for i in range(depth_images.shape[0]):
            depth_rgb = depth_images[i].squeeze(0).numpy()
            depth_rgb = (depth_rgb - depth_rgb.min()) / (
                depth_rgb.max() - depth_rgb.min()
            )
            colormap = plt.get_cmap("plasma")
            depth_rgb = colormap(depth_rgb)[:, :, :3]
            depth_rgb = torch.from_numpy(depth_rgb).permute(2, 0, 1)
            depth_rgb_batch.append(depth_rgb)
        depth_rgb_batch = torch.stack(depth_rgb_batch)

        return {
            "image": rgb_images,
            "depth": depth_images,
            "depth_rgb": depth_rgb_batch,
            "mask": mask_images,
            "xyz": xyzs,
            "bbox": bboxes,
            "rotation": self.rotations[obj_idx],
            "translation": self.translations[obj_idx],
            "obj_id": self.dataset_info["obj_ids"][obj_idx],
        }

    def get_dataloader(self, batch_size, shuffle, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    root_dir = "/export/datasets/public/3d/BOP/T-LESS"
    split = "train_pbr"  # "test_primesense"
    mode = "train"
    dataset = BOP_dataset(
        root_dir,
        split,
        mode,
        get_obj_gt_visible_masks=True,
        dataset_info=BOP_dataset_info["T-LESS"],
        image_name_template=BOP_dataset_info["T-LESS"]["image_name_template"],
        depth_name_template=BOP_dataset_info["T-LESS"]["depth_name_template"],
    )
    dataset.prepare_metadata()
    # dataset.get_obj_templates()
    # The code snippet `print(len(dataset))` is printing the length of the dataset, which is the number
    # of items in the metadata dictionary of the BOP_dataset instance.

    dl = dataset.get_dataloader(batch_size=2, shuffle=True)

    for i, data in enumerate(dl):
        scene_im_info = data["scene_im_info"][i].to_device("cuda")
        print(data["image"].shape)
        print(data["depth"].shape)
        print(data["scene_im_info"])
        if i == 0:
            break
