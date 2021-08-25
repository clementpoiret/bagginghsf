from functools import partial

import hydra
import pytorch_lightning as pl
import torchio as tio
from hydra import compose, initialize
from omegaconf import DictConfig
from torch import nn, optim

from bagginghsf.data.loader import load_from_config
from bagginghsf.models.losses import FocalTversky_loss
from bagginghsf.models.models import SegmentationModel

# initialize(config_path="conf")
# cfg = compose(config_name="config")
# print(OmegaConf.to_yaml(cfg))


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Load and setup data
    mri_datamodule = load_from_config(cfg.datasets)(
        preprocessing_pipeline=tio.Compose([
            tio.ToCanonical(),
            tio.ZNormalization(),
            tio.EnsureShapeMultiple(8),
        ]),
        augmentation_pipeline=tio.OneOf(
            {
                tio.RandomAffine(): .8,
            },
            p=1,
        ),
        postprocessing_pipeline=tio.Compose([tio.OneHot()]))
    # mri_datamodule.setup()

    # batch = next(iter(mri_datamodule.train_dataloader()))
    # batch["label"]["data"].shape

    # Training parameters
    seg_loss = FocalTversky_loss({"apply_nonlin": None})
    optimizer = optim.AdamW
    scheduler = partial(optim.lr_scheduler.CosineAnnealingLR,
                        T_max=cfg.lightning.max_epochs)
    learning_rate = 1e-3
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

    trainer = pl.Trainer(**cfg.lightning)

    trainer.fit(model, datamodule=mri_datamodule)


if __name__ == "__main__":
    main()
