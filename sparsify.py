# Comet must come first
from comet_ml import Experiment

# Import all other modules
import os
from functools import partial

import hydra
import pytorch_lightning as pl
import torch
import torchio as tio
from hydra import compose, initialize
from omegaconf import DictConfig
# from neptune.new.integrations.pytorch_lightning import NeptuneLogger
# from pytorch_lightning.callbacks import QuantizationAwareTraining
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.loggers import CometLogger
from torch import nn, optim

from bagginghsf.data.loader import load_from_config
from bagginghsf.models.losses import FocalTversky_loss
from bagginghsf.models.models import SegmentationModel

initialize(config_path="conf")
cfg = compose(config_name="config")
# print(OmegaConf.to_yaml(cfg))
VER = "1.1.0"


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # For multiple bags
    # Training parameters
    seg_loss = FocalTversky_loss({"apply_nonlin": None})
    optimizer = optim.AdamW
    scheduler = partial(optim.lr_scheduler.CosineAnnealingLR,
                        T_max=cfg.lightning.max_epochs)
    learning_rate = 1e-4
    # classes_names = None

    # Load and setup model
    if cfg.models.n == 1:
        model_name = list(cfg.models.models.keys())[0]
        hparams = cfg.models.models[model_name].hparams
        is_capsnet = cfg.models.models[model_name].is_capsnet
    else:
        raise NotImplementedError

    model = SegmentationModel(hparams=hparams,
                              seg_loss=seg_loss,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              learning_rate=learning_rate,
                              is_capsnet=is_capsnet)
    input_sample = torch.randn(1, 1, 16, 16, 16)
    model.to_onnx("sparsify.onnx", input_sample=input_sample)