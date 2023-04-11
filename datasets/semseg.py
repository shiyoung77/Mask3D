import logging
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
from random import randrange


import numpy
import torch
from datasets.random_cuboid import RandomCuboid

import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml
# from yaml import CLoader as Loader
from torch.utils.data import Dataset
from datasets.scannet200.scannet200_constants import SCANNET_COLOR_MAP_200, SCANNET_COLOR_MAP_20

logger = logging.getLogger(__name__)


class SemanticSegmentationDataset(Dataset):
    """Docstring for SemanticSegmentationDataset. """

    def __init__(
        self,
        dataset_name="scannet",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        instance_oversampling=0,
        place_around_existing=False,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        cache_data=False,
        add_unlabeled_pc=False,
        task="instance_segmentation",
        cropping=False,
        cropping_args=None,
        is_tta=False,
        crop_min_size=20000,
        crop_length=6.0,
        cropping_v1=True,
        reps_per_epoch=1,
        area=-1,
        on_crops=False,
        eval_inner_core=-1,
        filter_out_classes=[],
        label_offset=0,
        add_clip=False,
        is_elastic_distortion=True,
        color_drop=0.0
    ):
        assert task in ["instance_segmentation", "semantic_segmentation"], "unknown task"

        self.add_clip = add_clip
        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.color_drop = color_drop

        if self.dataset_name == "scannet":
            self.color_map = SCANNET_COLOR_MAP_20
            self.color_map[255] = (255, 255, 255)
        elif self.dataset_name == "scannet200":
            self.color_map = SCANNET_COLOR_MAP_200
        else:
            assert False, "dataset not known"

        self.task = task

        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset

        self.area = area
        self.eval_inner_core = eval_inner_core

        self.reps_per_epoch = reps_per_epoch  # 1

        self.cropping = cropping
        self.cropping_args = cropping_args
        self.is_tta = is_tta
        self.on_crops = on_crops

        self.crop_min_size = crop_min_size
        self.crop_length = crop_length

        self.version1 = cropping_v1

        self.random_cuboid = RandomCuboid(self.crop_min_size,
                                          crop_length=self.crop_length,
                                          version1=self.version1)

        self.mode = mode
        self.data_dir = data_dir
        self.add_unlabeled_pc = add_unlabeled_pc
        if add_unlabeled_pc:
            self.other_database = self._load_yaml(
                Path(data_dir).parent / "matterport" / "train_database.yaml"
            )
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.instance_oversampling = instance_oversampling
        self.place_around_existing = place_around_existing
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut
        self.flip_in_center = flip_in_center
        self.noise_rate = noise_rate
        self.resample_points = resample_points

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            if not (database_path / f"{mode}_database.yaml").exists():
                print(f"generate {database_path}/{mode}_database.yaml first")
                exit()
            self._data.extend(self._load_yaml(database_path / f"{mode}_database.yaml"))
        if data_percent < 1.0:
            self._data = sample(self._data, int(len(self._data) * data_percent))
        labels = self._load_yaml(Path(label_db_filepath))  # 200

        # if working only on classes for validation - discard others
        self._labels = self._select_correct_labels(labels, num_labels)  # 200

        if Path(str(color_mean_std)).exists():
            color_mean_std = self._load_yaml(color_mean_std)
            color_mean = tuple(color_mean_std["mean"])
            color_std = tuple(color_mean_std["std"])
        elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
            color_mean, color_std = color_mean_std[0], color_mean_std[1]
        else:
            logger.error("pass mean and std as tuple of tuples, or as an .yaml file")
            exit(1)

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (volume_augmentations_path != "none"):
            self.volume_augmentations = V.load(volume_augmentations_path, data_format="yaml")

        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (image_augmentations_path != "none"):
            self.image_augmentations = A.load(image_augmentations_path, data_format="yaml")

        # mandatory color augmentation
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        self.cache_data = cache_data  # False

    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)

    def __len__(self):
        return self.reps_per_epoch * len(self.data)

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)
        if self.is_tta:
            idx = idx % len(self.data)

        points = np.load(self.data[idx]["filepath"].replace("../../", ""))
        coordinates, color, normals, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
        )

        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        # volume and image augmentations for train
        if "train" in self.mode or self.is_tta:
            coordinates -= coordinates.mean(0)

            try:
                coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
            except OverflowError as err:
                print(coordinates)
                print(coordinates.shape)
                raise err

            for i in (0, 1):
                if random() < 0.5:
                    coord_max = np.max(points[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]

            if random() < 0.95:
                if self.is_elastic_distortion:
                    for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                        coordinates = elastic_distortion(
                            coordinates, granularity, magnitude
                        )
            aug = self.volume_augmentations(
                points=coordinates, normals=normals, features=color, labels=labels,
            )
            coordinates, color, normals, labels = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(self.image_augmentations(image=pseudo_image)["image"])

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        # prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            if not self.add_instance:  # self.add_instance is True
                # taking only first column, which is segmentation label, not instance
                labels = labels[:, 0].flatten()[..., None]

        labels = np.hstack((labels, segments[..., None].astype(np.int32)))

        features = color
        if self.add_normals:  # False
            features = np.hstack((features, normals))
        if self.add_raw_coordinates:  # True
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))

        if self.data[idx]['raw_filepath'].split("/")[-2] in ['scene0636_00', 'scene0154_00']:
            return self.__getitem__(0)

        return coordinates, features, labels, self.data[idx]['raw_filepath'].split("/")[-2], \
            raw_color, raw_normals, raw_coordinates, idx

    @property
    def data(self):
        """ database file containing information about preproscessed dataset """
        return self._data

    @property
    def label_info(self):
        """ database file containing information labels used by dataset """
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.load(f, Loader=yaml.FullLoader)
        return file

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for k, v, in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for k, v, in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def _remap_from_zero(self, labels):
        labels[~np.isin(labels, list(self.label_info.keys()))] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    def _remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud
