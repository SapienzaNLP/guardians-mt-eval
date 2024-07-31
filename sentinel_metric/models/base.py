r"""
RegressionMetricModel
========================
    Abstract Regression Metric Model class that implements some of the Pytorch Lightning logic.
    Extend this class to create a new regression metric model.
"""
import abc
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sentinel_metric.mt_metric import MTMetric

from .utils import (
    OrderedSampler,
    Prediction,
    Target,
    restore_list_order,
)

logger = logging.getLogger(__name__)


class RegressionMetricModel(pl.LightningModule, MTMetric):
    """RegressionMetricModel: Base class for all metric models. It uses MSE as loss.

    Args:
        name (str): The name of the metric. Defaults to None.
        optimizer (str): Optimizer used during training. Defaults to 'RAdam'.
        warmup_steps (int): Warmup steps for LR scheduler. Defaults to 0.
        learning_rate (float): Learning rate used to fine-tune the top layers. Defaults
            to 1e-05.
        batch_size (int): Batch size used during training. Defaults to 4.
        shuffle (bool): Flag that turns on and off the shuffle of the training data. Defaults to True.
        train_data (Optional[List[str]]): List of paths to training data. Defaults to None.
        validation_data (Optional[List[str]]): List of paths to validation data.
            Validation results are averaged across validation set. Defaults to None.
        class_identifier (Optional[str]): String used to identify the class of the model.
    """

    def __init__(
        self,
        name: str = None,
        optimizer: str = "RAdam",
        warmup_steps: int = 0,
        learning_rate: float = 1e-05,
        batch_size: int = 4,
        shuffle: bool = True,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        class_identifier: Optional[str] = None,
    ) -> None:
        pl.LightningModule.__init__(self)
        MTMetric.__init__(self, name)
        self.save_hyperparameters()

        # If not defined here, metrics will not live in the same device as our model.
        self.init_metrics()

    @abc.abstractmethod
    def read_training_data(self, path: Path) -> List[Dict]:
        """Abstract method that reads the training data.

        Args:
            path (Path): Path to the csv file containing the training data.

        Returns:
            List[Dict]: List of dictionaries containing training samples.
        """
        pass

    @abc.abstractmethod
    def read_validation_data(self, path: Path) -> List[Dict]:
        """Abstract method that reads the validation data.

        Args:
            path (Path): Path to the csv file containing the validation data.

        Returns:
            List[Dict]: List of dictionaries containing validation samples.
        """
        pass

    @abc.abstractmethod
    def prepare_sample(
        self,
        sample: List[Dict],
        stage: str = "fit",
    ):
        """This method will be called by dataloaders to prepared data to input to the
        model.

        Args:
            sample (List[Dict]): Batch of train/val/test samples.
            stage (str): model stage (options: 'fit', 'validate', 'test', or
                'predict'). Defaults to 'fit'.

        Returns:
            Model inputs and (optionally) training labels/targets.
        """
        pass

    @abc.abstractmethod
    def configure_optimizers(self):
        """Pytorch Lightning method to configure optimizers and schedulers."""
        pass

    @abc.abstractmethod
    def init_metrics(self) -> None:
        """Initializes metrics."""
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Prediction:
        """Pytorch model forward method."""
        pass

    @property
    def loss(self):
        """Loss function"""
        return torch.nn.MSELoss()

    def compute_loss(self, prediction: Prediction, target: Target) -> torch.Tensor:
        """Computes Loss value between a batch Prediction and respective Target."""
        return self.loss(prediction.score, target.score)

    def training_step(
        self,
        batch: Tuple[dict, Target],
        batch_idx: int,
    ) -> torch.Tensor:
        """Pytorch Lightning training step.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.

        Returns:
            [torch.Tensor] Loss value
        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        loss_value = self.compute_loss(batch_prediction, batch_target)

        self.log(
            "train_mse_loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_target.score.shape[0],
        )
        return loss_value

    def validation_step(
        self,
        batch: Tuple[dict, Target],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Pytorch Lightning validation step. Runs model and logs metrics.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.
            dataloader_idx (int): Integer displaying which dataloader this sample is
                coming from.

        """
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        mse_loss_value = self.compute_loss(batch_prediction, batch_target)

        self.val_metrics[dataloader_idx].update(
            batch_prediction.score, batch_target["score"], mse_loss=mse_loss_value
        )

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Prediction:
        """Pytorch Lightning predict step.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.
            dataloader_idx (int): Integer displaying which dataloader this sample is
                coming from.

        Return:
            Prediction object
        """

        return self.forward(**batch)

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        """Computes and logs metrics."""
        val_metrics = []
        for i in range(len(self.hparams.validation_data)):
            results = self.val_metrics[i].compute()
            self.val_metrics[i].reset()
            # Log the results for this validation set.
            self.log_dict(results, prog_bar=False)
            val_metrics.append(results)

        average_results = {
            "val_" + "_".join(k.split("_")[-2:]): [] for k in val_metrics[0].keys()
        }
        for i in range(len(val_metrics)):
            for k, v in val_metrics[i].items():
                average_results["val_" + "_".join(k.split("_")[-2:])].append(v)

        self.log_dict(
            {k: sum(v) / len(v) for k, v in average_results.items()}, prog_bar=True
        )

    def setup(self, stage: str) -> None:
        """Data preparation function called before training by Lightning.

        stage (str): either 'fit', 'validate', 'test', or 'predict'
        """
        if stage in (None, "fit"):
            self.validation_sets = [
                self.read_validation_data(d) for d in self.hparams.validation_data
            ]

    def train_dataloader(self) -> DataLoader:
        """Method that loads the train dataloader. Can be called every epoch to load a
        different trainset if `reload_dataloaders_every_n_epochs=1` in Lightning
        Trainer.
        """
        data_path = self.hparams.train_data[
            self.current_epoch % len(self.hparams.train_data)
        ]
        train_dataset = self.read_training_data(data_path)
        logger.info(f"Loading {data_path}. Shuffle: {self.hparams.shuffle}.")

        return DataLoader(
            dataset=train_dataset,
            sampler=RandomSampler(train_dataset)
            if self.hparams.shuffle
            else SequentialSampler(train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=lambda s: self.prepare_sample(s, stage="fit"),
            num_workers=2 * self.trainer.num_devices,
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation sets."""
        val_data = []
        for validation_set in self.validation_sets:
            val_data.append(
                DataLoader(
                    dataset=validation_set,
                    batch_size=self.hparams.batch_size,
                    collate_fn=lambda s: self.prepare_sample(s, stage="validate"),
                    num_workers=2 * self.trainer.num_devices,
                )
            )
        return val_data

    def prepare_for_inference(self, sample):
        return self.prepare_sample(sample, stage="predict")

    def predict(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 16,
        gpus: int = 1,
        devices: Union[List[int], str, int] = None,
        progress_bar: bool = True,
        accelerator: str = "auto",
        num_workers: int = None,
        length_batching: bool = True,
    ) -> Prediction:
        """Method that receives a list of samples (dictionaries with translations,
        sources and/or references) and returns segment-level scores and system level score.

        Args:
            samples (List[Dict[str, str]]): List with dictionaries with source,
                translations and/or references.
            batch_size (int): Batch size used during inference. Defaults to 16
            gpus: (int): Number of gpus to use. Defaults to 1.
            devices (Optional[List[int]]): A sequence of device indices to be used.
                Default: None.
            progress_bar (bool): Flag that turns on and off the predict progress bar.
                Defaults to True
            accelerator (str): Pytorch Lightning accelerator (e.g: 'cpu', 'cuda', 'hpu'
                , 'ipu', 'mps', 'tpu'). Defaults to 'auto'
            num_workers (int): Number of workers to use when loading and preparing
                data. Defaults to None
            length_batching (bool): If set to true, reduces padding by sorting samples
                by sequence length. Defaults to True.

        Return:
            Prediction object with `scores` and `system_score`.
        """
        if gpus > 1:
            raise ValueError(
                "Metrics predict method for the moment only supports at most one GPU."
            )
        if gpus == 1 and devices is not None:
            assert len(devices) == gpus, AssertionError(
                "List of devices must be same size as `gpus` or None if `gpus=0`"
            )
        elif gpus == 1:
            devices = gpus
        else:  # gpu = 0
            devices = "auto"

        sampler = SequentialSampler(samples)
        if length_batching and gpus < 2:
            if "src" in samples[0]:
                sent_to_use = "src"
            elif "mt" in samples[0]:
                sent_to_use = "mt"
            else:
                sent_to_use = "ref"
            sort_ids = np.argsort([len(sample[sent_to_use]) for sample in samples])
            sampler = OrderedSampler(sort_ids)

        if num_workers is None:
            num_workers = 2 * gpus  # just a rule of thumb

        self.eval()
        dataloader = DataLoader(
            dataset=samples,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.prepare_for_inference,
            num_workers=num_workers,
        )

        callbacks = []

        enable_progress_bar = progress_bar

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer = pl.Trainer(
            devices=devices,
            logger=False,
            callbacks=callbacks,
            accelerator=accelerator if gpus > 0 else "cpu",
            strategy="auto" if gpus < 2 else "ddp",
            enable_progress_bar=enable_progress_bar,
        )
        return_predictions = False if gpus > 1 else True
        predictions = trainer.predict(
            self, dataloaders=dataloader, return_predictions=return_predictions
        )

        scores = torch.cat([pred.score for pred in predictions], dim=0).tolist()
        output = Prediction(scores=scores, system_score=sum(scores) / len(scores))

        # Restore order of samples
        if length_batching and gpus < 2:
            output["scores"] = restore_list_order(scores, sort_ids)

        return output
