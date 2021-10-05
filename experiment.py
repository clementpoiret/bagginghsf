# Comet must come first
from comet_ml import Experiment

# Import all other modules
import os
from functools import partial

import hydra
import pytorch_lightning as pl
import torch
import torchio as tio
# from hydra import compose, initialize
from omegaconf import DictConfig
# from neptune.new.integrations.pytorch_lightning import NeptuneLogger
# from pytorch_lightning.callbacks import QuantizationAwareTraining
from pytorch_lightning.loggers import CometLogger
from torch import nn, optim

from bagginghsf.data.loader import load_from_config
from bagginghsf.models.losses import FocalTversky_loss
from bagginghsf.models.models import SegmentationModel

# initialize(config_path="conf")
# cfg = compose(config_name="config")
# print(OmegaConf.to_yaml(cfg))
VER = "1.1.0"


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # For multiple bags
    for i in range(cfg.models.n_models):
        # Logger
        logger = CometLogger(**cfg.logger)

        # Load and setup data
        mri_datamodule = load_from_config(cfg.datasets)(
            preprocessing_pipeline=tio.Compose([
                tio.ToCanonical(),
                tio.ZNormalization(),
            ]),
            augmentation_pipeline=tio.Compose([
                tio.RandomFlip(axes=('LR',), p=.5),
                tio.RandomMotion(degrees=5,
                                 translation=5,
                                 num_transforms=3,
                                 p=.1),
                tio.RandomBlur(std=(0, 0.5), p=.1),
                tio.RandomNoise(mean=0, std=0.5, p=.1),
                tio.RandomGamma(log_gamma=0.4, p=.1),
                tio.RandomAffine(scales=.3,
                                 degrees=30,
                                 translation=5,
                                 isotropic=False,
                                 p=.2),
                # tio.RandomAnisotropy(p=.1, scalars_only=False),
                tio.transforms.RandomElasticDeformation(num_control_points=4,
                                                        max_displacement=4,
                                                        locked_borders=0,
                                                        p=.1),
                # tio.RandomSpike(p=.01),
                # tio.RandomBiasField(coefficients=.2, p=.01),
            ]),
            postprocessing_pipeline=tio.Compose(
                [tio.EnsureShapeMultiple(8),
                 tio.OneHot()]))
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

        # print("cwd:", os.getcwd())
        trainer = pl.Trainer(logger=logger, **cfg.lightning)

        # print("NUMBER OF GPUs:", torch.cuda.device_count())

        trainer.fit(model, datamodule=mri_datamodule)

        # torch.save(model.state_dict(), "unet_test.pt")
        trainer.save_checkpoint(f"arunet_{VER}_bag{i}.ckpt")
        # logger.experiment['model_checkpoints/arunet_c'].upload('arunet_v0c.ckpt')
        logger.experiment.log_model(f"arunet_{VER}_bag{i}_ckpt",
                                    f"arunet_{VER}_bag{i}.ckpt")

        dummy_input = torch.randn(1, 1, 16, 16, 16)
        model.eval()
        torch.onnx.export(model,
                          dummy_input,
                          f'arunet_{VER}_bag{i}.onnx',
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={
                              'input': {
                                  0: 'batch',
                                  2: "x",
                                  3: "y",
                                  4: "z"
                              },
                              'output': {
                                  0: 'batch',
                                  2: "x",
                                  3: "y",
                                  4: "z"
                              }
                          },
                          opset_version=13)
        logger.experiment.log_model(f"arunet_{VER}_bag{i}_onnx",
                                    f"arunet_{VER}_bag{i}.onnx")
        # torch.onnx.export(model.quant,
        #                   dummy_input,
        #                   f'arunet_{VER}_bag{i}_quant.onnx',
        #                   input_names=['input'],
        #                   output_names=['output'],
        #                   dynamic_axes={
        #                       'input': {
        #                           0: 'batch',
        #                           2: "x",
        #                           3: "y",
        #                           4: "z"
        #                       },
        #                       'output': {
        #                           0: 'batch',
        #                           2: "x",
        #                           3: "y",
        #                           4: "z"
        #                       }
        #                   })
        # logger.experiment.log_model(f"arunet_{VER}_bag{i}_quant_onnx",
        #                             f"arunet_{VER}_bag{i}_quant.onnx")

    # trainer.test(model, datamodule=mri_datamodule)
    # logger.experiment.stop()


if __name__ == "__main__":
    main()
