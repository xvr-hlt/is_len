#!/usr/bin/env python
import copy
import logging
from typing import Tuple

import hydra
import omegaconf
import pytorch_lightning as pl
import wandb


def load(config: omegaconf.DictConfig,) -> Tuple[pl.LightningModule, pl.LightningDataModule, pl.Trainer]:
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    logging.info(f"Instantiating experiment <{config.experiment._target_}>.")
    experiment: pl.LightningModule = hydra.utils.instantiate(config.experiment)

    logging.info(f"Instantiating datamodule <{config.datamodule._target_}>..")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    logging.info(f"Instantiating logger <{config.logger._target_}>.")
    logger: pl.loggers.LightningLoggerBase = hydra.utils.instantiate(config.logger)

    # this can be removed once https://github.com/PyTorchLightning/pytorch-lightning/issues/9264 is accepted.
    @pl.utilities.rank_zero_only
    def save_config():
        logger.experiment.config.update(omegaconf.OmegaConf.to_container(config, resolve=True))

    logging.info(f"Saving config.")
    save_config()

    logging.info(f"Instantiating trainer <{config.trainer._target_}>.")
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, logger=logger)
    return experiment, datamodule, trainer


@hydra.main(config_path="../config/", config_name="base.yaml")
def main(config: omegaconf.DictConfig):
    experiment, datamodule, trainer = load(config)
    logging.info(f"Running fit.")
    trainer.fit(experiment, datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()
