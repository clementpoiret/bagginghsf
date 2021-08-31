# Comet must come first
# from comet_ml import Experiment

# Import all other modules
import os
from functools import partial

import hydra
import pytorch_lightning as pl
import torch
import torchio as tio
# from hydra import compose, initialize
from omegaconf import DictConfig
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from torch import nn, optim

from bagginghsf.data.loader import load_from_config
from bagginghsf.models.losses import FocalTversky_loss
from bagginghsf.models.models import SegmentationModel

# initialize(config_path="conf")
# cfg = compose(config_name="config")
# print(OmegaConf.to_yaml(cfg))


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Logger
    logger = NeptuneLogger(**cfg.logger)

    # Load and setup data
    mri_datamodule = load_from_config(cfg.datasets)(
        preprocessing_pipeline=tio.Compose([
            tio.ToCanonical(),
            tio.ZNormalization(),
            tio.EnsureShapeMultiple(8),
        ]),
        augmentation_pipeline=tio.Compose([
            # tio.transforms.RandomElasticDeformation(num_control_points=5,
            #                                         max_displacement=3,
            #                                         locked_borders=2,
            #                                         p=.05),
            tio.RandomFlip(axes=('LR',), flip_probability=.2),
            # tio.RandomAffine(scales=.5, degrees=10, translation=3, p=.1),
            # tio.RandomMotion(degrees=5, translation=5, num_transforms=2, p=.01),
            # tio.RandomSpike(p=.01),
            # tio.RandomBiasField(coefficients=.2, p=.01),
            tio.RandomBlur(p=.01),
            tio.RandomNoise(p=.1),
            tio.RandomGamma(p=.1),
        ]),
        postprocessing_pipeline=tio.Compose([tio.OneHot()]))
    # mri_datamodule.setup()

    # batch = next(iter(mri_datamodule.train_dataloader()))
    # batch["label"]["data"].shape

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

    print("cwd:", os.getcwd())
    trainer = pl.Trainer(logger=logger, **cfg.lightning)

    trainer.fit(model, datamodule=mri_datamodule)

    torch.save(model.state_dict(), "unet_test.pt")
    logger.experiment['model_checkpoints/unet_test'].upload('unet_test.pt')
    # logger.experiment.log_model("poc", "unet_test.pt")

    trainer.test(model, datamodule=mri_datamodule)
    logger.experiment.stop()


if __name__ == "__main__":
    main()
