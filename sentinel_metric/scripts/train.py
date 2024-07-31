import json
import logging
import warnings
from pathlib import Path

from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer

from sentinel_metric.models import SentinelRegressionMetric

logger = logging.getLogger(__name__)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for training metric models.")
    parser.add_argument(
        "--seed-everything",
        type=int,
        default=12,
        help="Training Seed.",
    )
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_subclass_arguments(SentinelRegressionMetric, "sentinel_regression_metric")
    parser.add_subclass_arguments(ModelCheckpoint, "model_checkpoint")
    parser.add_subclass_arguments(WandbLogger, "wandb_logger")
    parser.add_argument("--wandb-logger-entity", type=str, help="Wandb entity name.")
    parser.add_subclass_arguments(Trainer, "trainer")
    parser.add_argument(
        "--load-from-checkpoint",
        type=str,
        help="Loads a model checkpoint for fine-tuning",
        default=None,
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Strictly enforce that the keys in checkpoint_path match the keys returned by this module's state dict.",
    )
    return parser


def initialize_trainer(configs) -> Trainer:
    checkpoint_callback = ModelCheckpoint(
        **namespace_to_dict(configs.model_checkpoint.init_args)
    )
    wandb_logger_args = namespace_to_dict(configs.wandb_logger.init_args)
    wandb_logger_args["entity"] = configs.wandb_logger_entity
    wandb_logger = WandbLogger(**wandb_logger_args)
    trainer_args = namespace_to_dict(configs.trainer.init_args)
    rich_progress_bar_callback = RichProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer_args["callbacks"] = [
        rich_progress_bar_callback,
        checkpoint_callback,
        lr_monitor,
    ]
    trainer_args["logger"] = wandb_logger
    trainer = Trainer(**trainer_args)
    return trainer


def initialize_model(configs):
    print("MODEL ARGUMENTS: ")

    if configs.sentinel_regression_metric is not None:
        print(
            json.dumps(
                configs.sentinel_regression_metric.init_args,
                indent=4,
                default=lambda x: x.__dict__,
            )
        )
        if configs.load_from_checkpoint is not None:
            logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
            model = SentinelRegressionMetric.load_from_checkpoint(
                checkpoint_path=configs.load_from_checkpoint,
                strict=configs.strict_load,
                **namespace_to_dict(configs.sentinel_regression_metric.init_args),
            )
        else:
            model = SentinelRegressionMetric(
                **namespace_to_dict(configs.sentinel_regression_metric.init_args)
            )
    else:
        raise Exception("Model configurations missing!")

    return model


def train_command() -> None:
    parser = read_arguments()
    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything, workers=True)

    trainer = initialize_trainer(cfg)
    model = initialize_model(cfg)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*Consider increasing the value of the `num_workers` argument` .*",
    )
    trainer.fit(model)


if __name__ == "__main__":
    train_command()
