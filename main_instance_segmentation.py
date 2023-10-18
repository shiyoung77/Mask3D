import logging
import os
import hydra
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer, seed_everything


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = []
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))
    callbacks.append(RegularCheckpointing())

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
        ckpt_path = None
    else:
        print("EXPERIMENT ALREADY EXIST")
        ckpt_path = Path(f"{cfg.general.save_dir}/last-epoch.ckpt")
        if not ckpt_path.exists():
            ckpt_path = None

    runner = Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=cfg.general.gpus,
        callbacks=callbacks,
        **cfg.trainer,
    )
    # ckpt_path = "checkpoints/scannet200_val.ckpt"
    runner.fit(model, ckpt_path=ckpt_path)
    # runner.fit(model)


def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        accelerator="gpu",
        devices=cfg.general.gpus,
        logger=loggers,
        **cfg.trainer
    )
    runner.validate(model)


def debug(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        accelerator="gpu",
        devices=cfg.general.gpus,
        logger=loggers,
        **cfg.trainer
    )
    runner.predict(model)


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg: DictConfig):
    if cfg['general']['train_mode']:
        train(cfg)
    else:
        # test(cfg)
        debug(cfg)


if __name__ == "__main__":
    main()
