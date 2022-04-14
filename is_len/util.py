import functools

import hydra
import torch


def partial(_partial_: str, *args, **kwargs):
    """Helper method for defining `partial`s in the context of a hydra config file.
    Args:
        _partial_ (str): Name of object to make a `partial` of.
    """
    return functools.partial(hydra.utils.get_method(_partial_), *args, **kwargs)


def get_config(config_path="../config", config_name="base.yaml", overrides=None):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path, job_name="debug")
    return hydra.compose(config_name=config_name, overrides=(overrides or []))


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556],
          [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
