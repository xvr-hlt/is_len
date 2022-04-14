from typing import Callable, Dict, List, Optional

import pytorch_lightning as pl
import wandb
from torch import nn


class LeninDetector(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 optim_constructor: nn.Module,
                 optim_scheduler_constructor: nn.Module,
                 log_n_images: int = 4,
                 log_top_n_boxes: int = 10,
                 img_preprocessor: Optional[Callable] = None):
        super().__init__()
        self.model = model
        self.optim_constructor = optim_constructor
        self.optim_scheduler_constructor = optim_scheduler_constructor
        self.log_n_images = log_n_images
        self.log_top_n_boxes = log_top_n_boxes

    def forward(self, *args, **input):
        return self.model(*args, **input)

    def step(self, batch: Dict, step_prefix: str, log_images: Optional[int] = None):
        paths = batch.pop('paths')
        output = self(**batch)
        self.log(f"{step_prefix}_loss", output.loss)
        self.log_dict({f"{step_prefix}_{k}": v for k, v in output.loss_dict.items()})
        if log_images:
            self.log_images(batch, output, log_ixs=log_images, log_prefix=step_prefix, paths=paths)
        return output.loss

    def training_step(self, batch: Dict, batch_ix: int):  #type: ignore
        return self.step(batch, 'train', self.log_n_images)

    def validation_step(self, batch: Dict, batch_ix: int):  #type: ignore
        return self.step(batch, 'val', self.log_n_images)

    def predict_step(self, batch: Dict, batch_idx: int, dataloader_idx: Optional[int] = None):
        pass

    def configure_optimizers(self):
        optim = self.optim_constructor((p for p in self.model.parameters() if p.requires_grad))
        optim_scheduler = self.optim_scheduler_constructor(optimizer=optim)
        return {"optimizer": optim, "lr_scheduler": optim_scheduler}

    def log_images(self, batch, output, log_ixs: int, log_prefix: str, paths: List[str]):
        wandb_images = []
        boxes_for_each_image = []
        captions = []
        for ix in range(min(log_ixs, len(batch['labels']))):
            img = batch['pixel_values'][ix]
            probs = output.logits[ix].softmax(-1)
            prob_some_box = (1 - probs[:, -1])
            _, top_n_ixs = prob_some_box.topk(self.log_top_n_boxes)

            values, indices = probs[top_n_ixs].max(-1)
            boxes = output.pred_boxes[ix][top_n_ixs]
            box_data_pred = []
            # predictions
            for box_val, box_ix, (xc, yc, w, h) in zip(values.tolist(), indices.tolist(), boxes.tolist()):
                box_data_pred.append({
                    "position": {
                        "middle": [xc, yc],
                        "width": w,
                        "height": h
                    },
                    "class_id": int(box_ix),
                    "scores": {
                        "prob": box_val
                    }
                })
            # targets
            box_data_targets = []
            target_box, target_ixs = batch['labels'][ix]['boxes'], batch['labels'][ix]['class_labels']
            for (xc, yc, w, h), box_ix in zip(target_box.tolist(), target_ixs.tolist()):
                box_data_targets.append({
                    "position": {
                        "middle": [xc, yc],
                        "width": w,
                        "height": h
                    },
                    "class_id": int(box_ix),
                })
            boxes = {"predictions": {"box_data": box_data_pred}, "targets": {"box_data": box_data_targets}}
            wandb_images.append(wandb.Image(img))
            boxes_for_each_image.append(boxes)
            captions.append(paths[ix])
        self.logger.log_image(key=f"{log_prefix}_bounding_boxes",
                              images=wandb_images,
                              boxes=boxes_for_each_image,
                              step=self.current_epoch,
                              caption=captions)
