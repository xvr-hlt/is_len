import albumentations as A
import cv2
import numpy as np


def get_torch_transform(transform):
    # albumentations
    def torch_transform(img, bboxes, class_labels):
        img = np.array(img)
        transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        return [transformed[k] for k in ('image', 'bboxes', 'class_labels')]

    return torch_transform


def get_basic_transform():
    return get_torch_transform(
        transform=A.Compose([
            A.RandomSizedBBoxSafeCrop(width=512, height=512, erosion_rate=0.2, p=.5),
            A.HorizontalFlip(p=0.25),
            A.RandomBrightnessContrast(p=0.5),
            A.CLAHE(),
            A.MotionBlur(blur_limit=5, p=.25),
            A.Cutout()
        ],
                            bbox_params=A.BboxParams(
                                format='albumentations', min_visibility=0.5, label_fields=['class_labels'])))
