import copy
import pathlib
from typing import Union

import fire
import hydra
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
import tqdm.auto as tqdm
from loguru import logger
from torch import nn

NON_LENIN_IX = 0
LENIN_IX = 1


def get_experiment_from_checkpoint(experiment_config: omegaconf.DictConfig, checkpoint: str) -> pl.LightningModule:
    """Get pytorch LightningModule from checkpoint.
    Args:
        experiment_config (omegaconf.DictConfig): Config for experiment.
        checkpoint (str): Checkpoint path.
    Returns:
        pl.LightningModule
    """
    experiment_config = copy.deepcopy(experiment_config)
    target = experiment_config._target_
    del experiment_config._target_
    kwargs = hydra.utils.instantiate(experiment_config)
    experiment: pl.LightningModule = hydra.utils.get_class(target).load_from_checkpoint(checkpoint,
                                                                                        **kwargs)  # type: ignore
    return experiment


def cxcywh_to_x0y0x1y1(bbox: torch.Tensor):
    cx, cy, w, h = bbox.tolist()
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def get_iou(bbox0, bbox1):
    ax0, ay0, ax1, ay1 = bbox0
    bx0, by0, bx1, by1 = bbox1

    # determine the coordinates of the intersection rectangle
    x_left = max(ax0, bx0)
    y_top = max(ay0, by0)
    x_right = min(ax1, bx1)
    y_bottom = min(ay1, by1)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (ax1 - ax0) * (ay1 - ay0)
    bb2_area = (bx1 - bx0) * (by1 - by0)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def get_box_cls_from_predictions(pred_cls: torch.Tensor, boxes: torch.Tensor, probs: torch.Tensor):
    if (pred_cls == LENIN_IX).any():  # if any lenin, choose most lenin.
        predicted_lenin_mask = (pred_cls == LENIN_IX)
        box_ix = probs[predicted_lenin_mask].argmax()
        box, cls = cxcywh_to_x0y0x1y1(boxes[predicted_lenin_mask][box_ix]), 1
    elif (pred_cls == NON_LENIN_IX).any():  # if any non-lenin, choose biggest area
        w, h = boxes[:, -2], boxes[:, -1]
        area = w * h
        predicted_non_lenin_ix = (pred_cls == NON_LENIN_IX)
        box_ix = area[predicted_non_lenin_ix].argmax()
        box, cls = cxcywh_to_x0y0x1y1(boxes[predicted_non_lenin_ix][box_ix]), 0
    else:
        box, cls = [0, 0, 1, 1], 0
    return box, cls


@torch.no_grad()
def infer(input_folder: str,
          model_path: Union[pathlib.Path, str] = "model/20220225_18w2yw4m",
          output_path: str = "results.csv",
          checkpoint_name: str = "model.ckpt",
          config_name: str = "base.yaml"):
    model_path = pathlib.Path(model_path)
    model_conf: omegaconf.DictConfig = omegaconf.OmegaConf.load(model_path / config_name)  # type: ignore
    logger.info(f"Loading model.")
    model = get_experiment_from_checkpoint(model_conf.experiment, str(model_path / checkpoint_name))
    model.eval().cuda()
    datamodule = hydra.utils.instantiate(model_conf.datamodule, predict_home=input_folder)
    dataloader = datamodule.predict_dataloader()
    results = []
    for batch in tqdm.tqdm(dataloader):
        paths = batch.pop('paths')
        outputs = model(**{k: batch[k].cuda() for k in ('pixel_values', 'pixel_mask')})
        values, indices = outputs.logits.softmax(-1).max(-1)
        for path, probs, pred_cls, boxes in zip(paths, values, indices, outputs['pred_boxes']):
            box, cls = get_box_cls_from_predictions(pred_cls, boxes, probs)
            path_stub = pathlib.Path(path).parts[-1]
            logger.info(f"{path_stub}, {box}, {cls}")
            box_str = ';'.join([str(point) for point in box])
            results.append(f"{path_stub};{box_str};{cls}")

    with open(output_path, "w") as f:
        f.write('\n'.join(results))
    return results


def evaluate(model_path: Union[pathlib.Path, str] = "model/20220225_18w2yw4m",
             checkpoint_name: str = "model.ckpt",
             config_name: str = "base.yaml",
             return_all_results: bool = False):
    model_path = pathlib.Path(model_path)
    model_conf: omegaconf.DictConfig = omegaconf.OmegaConf.load(model_path / config_name)  # type: ignore
    logger.info(f"Loading model.")
    model = get_experiment_from_checkpoint(model_conf.experiment, str(model_path / checkpoint_name))
    model.eval().cuda()
    datamodule = hydra.utils.instantiate(model_conf.datamodule)
    datamodule.setup()
    results = []
    for batch in tqdm.tqdm(datamodule.val_dataloader()):
        paths = batch.pop('paths')
        outputs = model(**{k: batch[k].cuda() for k in ('pixel_values', 'pixel_mask')})
        values, indices = outputs.logits.softmax(-1).max(-1)
        for path, probs, pred_cls, boxes, labels in zip(paths, values, indices, outputs['pred_boxes'], batch['labels']):
            pred_box, pred_cls = get_box_cls_from_predictions(pred_cls, boxes, probs)
            true_box, true_cls = get_box_cls_from_predictions(labels["class_labels"], labels["boxes"],
                                                              torch.ones_like(labels["class_labels"]))
            results.append({
                'path': path,
                'pred_box': pred_box,
                'pred_cls': pred_cls,
                'true_box': true_box,
                'true_cls': true_cls,
                'iou': get_iou(pred_box, true_box)
            })

    results_df = pd.DataFrame(results)
    results_summary = {
        f'micro_acc': (results_df.true_cls == results_df.pred_cls).mean(),
        f'micro_avg_iou': (results_df.iou).mean(),
        f'micro_iou>0.5': (results_df.iou > 0.5).mean(),
        f'micro_support': len(results_df)
    }
    for cls_ix, cls_name in enumerate(('other', 'lenin')):
        results_subset = results_df[results_df.true_cls == cls_ix]
        results_summary.update({
            f'cls_{cls_name}_acc': (results_subset.true_cls == results_subset.pred_cls).mean(),
            f'cls_{cls_name}_avg_iou': (results_subset.iou).mean(),
            f'cls_{cls_name}_iou>0.5': (results_subset.iou > 0.5).mean(),
            f'cls_{cls_name}_support': len(results_subset)
        })
    if return_all_results:
        return {'results': results, 'results_summary': results_summary}
    return results_summary


if __name__ == "__main__":
    fire.Fire({'infer': infer, 'evaluate': evaluate})
