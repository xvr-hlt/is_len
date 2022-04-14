from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import gdown
import numpy as np
import pandas as pd
import PIL
import pytorch_lightning
import torch
import torchvision
import transformers
from loguru import logger

import is_len
from is_len import datamodule

IMAGE_URL = "https://drive.google.com/u/0/uc?id=1NpTPzQvkAyh9YPS8v_i7z_EZjRy9lYGs&export=download"
LABEL_URL = "https://drive.google.com/file/d/18idZeW3IS1aUqXMypltL_WpZxa0C5opf/view?usp=sharing"
CLS_TO_IX = {'other': 0, 'lenin': 1}
DATA_HOME_DEFAULT = pathlib.Path(is_len.__file__).parent.parent / "data"


@dataclasses.dataclass
class BoundingBox:
    cls: str
    x0_y0_x1_y1: Tuple[float, float, float, float]


@dataclasses.dataclass
class Instance:
    path: pathlib.Path
    bboxes: List[BoundingBox]


@dataclasses.dataclass
class ImageBoxDataset(torch.utils.data.Dataset):
    instances: List[datamodule.Instance]
    max_size: int = 768
    augment: Optional[Callable] = None

    @staticmethod
    def flip_image(img, orientation_exif_key: int = 274) -> PIL.Image:
        exif = img._exif
        if exif is None or orientation_exif_key not in exif:
            return img
        orientation = exif[orientation_exif_key]
        rotate = {3: PIL.Image.ROTATE_180, 8: PIL.Image.ROTATE_90, 6: PIL.Image.ROTATE_270}
        if orientation in rotate:
            img = img.transpose(rotate[orientation])
        return img

    def __getitem__(self, ix: int) -> Dict[str, Any]:
        instance = self.instances[ix]
        img = PIL.Image.open(instance.path)
        img = self.flip_image(img)
        boxes = np.array([box.x0_y0_x1_y1 for box in instance.bboxes])
        class_labels = [CLS_TO_IX[box.cls] for box in instance.bboxes]

        if self.augment:
            img_np, aug_boxes, aug_class_labels = self.augment(np.array(img), boxes, class_labels)
            if aug_class_labels:
                img = PIL.Image.fromarray(img_np)
                boxes = aug_boxes
                class_labels = aug_class_labels

        img = img.resize((self.max_size, self.max_size), resample=PIL.Image.LANCZOS)
        img_tensor = torchvision.transforms.functional.to_tensor(img)

        boxes_xy_wh = torch.tensor([[(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0] for (x0, y0, x1, y1) in boxes
                                   ]).float()
        return {
            'img': img_tensor,
            'labels': {
                'boxes': boxes_xy_wh,
                'class_labels': torch.tensor(class_labels)
            },
            'paths': str(instance.path)
        }

    def __len__(self) -> int:
        return len(self.instances)


def get_collate_fn(feature_extractor=None):
    if feature_extractor is None:
        feature_extractor = transformers.DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50",
                                                                              do_resize=True,
                                                                              padding=True)

    def collate_fn(batch):
        pixel_values = [item['img'] for item in batch]
        encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        batch = {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': [item['labels'] for item in batch],
            'paths': [item['paths'] for item in batch]
        }
        return batch

    return collate_fn


class DataModule(pytorch_lightning.LightningDataModule):

    def __init__(self,
                 data_home: Optional[str] = None,
                 val_split_pct: float = 0.2,
                 dataset_train_kwargs: Optional[Dict] = None,
                 dataset_val_kwargs: Optional[Dict] = None,
                 dataloader_train_kwargs: Optional[Dict] = None,
                 dataloader_val_kwargs: Optional[Dict] = None,
                 predict_home: Optional[str] = None):
        self.data_home = pathlib.Path(data_home) if data_home is not None else DATA_HOME_DEFAULT
        self.train_instances: Optional[List[Instance]] = None
        self.val_instances: Optional[List[Instance]] = None
        self.val_split_pct = val_split_pct
        self.dataset_train_kwargs = dataset_train_kwargs
        self.dataset_val_kwargs = dataset_val_kwargs
        self.dataloader_train_kwargs = dataloader_train_kwargs
        self.dataloader_val_kwargs = dataloader_val_kwargs
        self.predict_home = predict_home

    def prepare_data(self) -> None:
        if not self.data_home.exists():
            self.data_home.mkdir(parents=True)
            zip_path = str(self.data_home / "statues-train.zip")
            gdown.download(IMAGE_URL, zip_path)
            gdown.download(LABEL_URL, str(self.data_home / "labels.csv"), fuzzy=True)
            gdown.extractall(zip_path)
            pathlib.Path(zip_path).unlink()

    def setup(self, stage: Optional[str] = None) -> None:
        labels = pd.read_csv(self.data_home / "labels.csv")
        instances: List[Instance] = []
        prev_filename = None
        bboxes: List[BoundingBox] = []
        path: pathlib.Path = None  # type: ignore

        for _, row in labels.iterrows():
            filename = row['filename']
            if (prev_filename is not None) and filename != prev_filename:
                instances.append(Instance(path=path, bboxes=bboxes))
                bboxes = []
            prev_filename = filename
            cls = row['class']
            path = self.data_home / "statues-train" / f"statues-{cls}" / filename
            if not path.exists():
                other_cls = {"lenin": "other", "other": "lenin"}[cls]
                alt_path = self.data_home / "statues-train" / f"statues-{other_cls}" / filename
                if not alt_path.exists():
                    logger.warning(f"Could not find {path=} in either folder")
                    continue
                path = alt_path

            width = row['width']
            height = row['height']
            x0, y0, x1, y1 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            bboxes.append(BoundingBox(cls=cls, x0_y0_x1_y1=(x0 / width, y0 / height, x1 / width, y1 / height)))

        instances = sorted(instances, key=lambda i: hashlib.md5(str(i.path).encode()).hexdigest())
        split_point = round(len(instances) * self.val_split_pct)
        self.val_instances, self.train_instances = instances[:split_point], instances[split_point:]
        logger.info(f"{len(self.val_instances)=}, {len(self.train_instances)=}")

    def train_dataloader(self):
        box_dataset = ImageBoxDataset(self.train_instances, **(self.dataset_train_kwargs or {}))
        return torch.utils.data.DataLoader(box_dataset, **self.dataloader_train_kwargs)

    def val_dataloader(self):
        box_dataset = ImageBoxDataset(self.val_instances, **(self.dataset_val_kwargs or {}))
        return torch.utils.data.DataLoader(box_dataset, **self.dataloader_val_kwargs)

    def predict_dataloader(self):
        if self.predict_home is None:
            return super().predict_dataloader()
        predict_instances = [Instance(path=path, bboxes=[]) for path in pathlib.Path(self.predict_home).iterdir()]
        box_dataset = ImageBoxDataset(predict_instances, **(self.dataset_val_kwargs or {}))
        return torch.utils.data.DataLoader(box_dataset, **self.dataloader_val_kwargs)
