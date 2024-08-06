r"""
SentinelRegressionMetric
================
    Sentinel Regression Metric that learns to predict a quality assessment by looking
    at incomplete MT Eval info.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal

import torch
from torch import nn
from torch.optim import RAdam
from transformers.optimization import Adafactor, get_constant_schedule_with_warmup

from sentinel_metric.encoders import str2encoder
from sentinel_metric.models.base import RegressionMetricModel
from sentinel_metric.models.utils import Prediction, Target, read_csv_data
from sentinel_metric.models.metrics import RegressionMetrics
from sentinel_metric.modules import FeedForward


class SentinelRegressionMetric(RegressionMetricModel):
    """SentinelRegressionMetric:

    Args:
        name (str): The name of the metric. Defaults to None.
        optimizer (str): Optimizer used during training. Defaults to 'RAdam'.
        warmup_steps (int): Warmup steps for LR scheduler. Defaults to 0.
        learning_rate (float): Learning rate used to fine-tune the top layers. Defaults
            to 1e-05.
        batch_size (int): Batch size used during training. Defaults to 4.
        shuffle (bool): Flag that turns on and off the shuffle of the training data. Defaults to True.
        train_data (Optional[List[str]]): List of paths to training data. Defaults to None.
        validation_data (Optional[List[str]]): List of paths to validation data. Validation results are averaged across
                                               validation set. Defaults to None.
        keep_embeddings_frozen (bool): Keeps the encoder frozen during training. Defaults to False.
        encoder_model (str): Encoder model to be used. Defaults to 'XLM-RoBERTa'.
        pretrained_model (str): Pretrained model from Hugging Face. Defaults to 'xlm-roberta-large'.
        load_pretrained_weights (Bool): If set to False it avoids loading the weights of the pretrained model
                                        (e.g. XLM-R) before it loads the metric model checkpoint. Defaults to True.
        dropout (float): Dropout used in the top-layers. Defaults to 0.1.
        hidden_sizes (List[int]): Hidden sizes for the Feed Forward regression.
        activations (str): Feed Forward activation function.
        sent_to_use (Literal["src", "mt", "ref"]): Which sentence to use in the fake metric. It must be in ["src", "mt",
                                                   "ref"].
        final_activation (str): Feed Forward final activation.
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
        keep_embeddings_frozen: bool = False,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-large",
        load_pretrained_weights: bool = True,
        dropout: float = 0.1,
        hidden_sizes: List[int] = None,
        activations: str = "Tanh",
        sent_to_use: Literal["src", "mt", "ref"] = "mt",
        final_activation: Optional[str] = None,
    ) -> None:
        if sent_to_use not in ["src", "mt", "ref"]:
            raise ValueError(
                "Input parameter 'sent_to_use' for 'SentinelRegressionMetric' class constructor must be 'src'"
                ", 'mt', or 'ref'!"
            )

        super().__init__(
            name=name,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            shuffle=shuffle,
            train_data=train_data,
            validation_data=validation_data,
            class_identifier="sentinel_regression_metric",
        )
        self.save_hyperparameters()

        self.encoder = str2encoder[self.hparams.encoder_model].from_pretrained(
            self.hparams.pretrained_model, load_pretrained_weights
        )
        if self.hparams.keep_embeddings_frozen:
            self.encoder.freeze_embeddings()

        self.estimator = FeedForward(
            in_dim=self.encoder.output_units,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
            out_dim=1,
        )

    def init_metrics(self):
        """Initializes metrics."""
        self.val_metrics = nn.ModuleList(
            [
                RegressionMetrics(
                    prefix=d.replace(".csv", "").replace("data", "val_metrics"),
                    compute_correlations=self.hparams.sent_to_use == "mt",
                )
                for d in self.hparams.validation_data
            ]
        )

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Pytorch Lightning method to configure optimizers and schedulers."""

        if self.hparams.optimizer == "RAdam":
            optimizer = RAdam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate
            )
        else:
            optimizer = Adafactor(
                self.parameters(),
                lr=self.hparams.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )

        if self.hparams.warmup_steps < 2:
            return [optimizer], []

        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
        )
        return [optimizer], [scheduler]

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "train"
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """This method will be called by dataloaders to prepared data to input to the model.

        Args:
            sample (List[dict]): Batch of train/val/test samples.
            stage (str): model stage (options: 'fit', 'validate', 'test', or
                'predict'). Defaults to 'fit'.

        Returns:
            Model inputs and depending on the 'stage' training labels/targets.
        """
        model_inputs = self.encoder.prepare_sample(
            [str(dic[self.hparams.sent_to_use]) for dic in sample]
        )

        if stage == "predict":
            return model_inputs

        scores = [float(s["score"]) for s in sample]
        targets = Target(score=torch.tensor(scores, dtype=torch.float))

        return model_inputs, targets

    def estimate(
        self,
        sentemb: torch.Tensor,
    ) -> Prediction:
        """Method that takes the sentence embeddings from the Encoder and runs the Estimator Feed-Forward on top.

        Args:
            mt_sentemb [torch.Tensor]: Sentence embedding that will be the input of the Feed-Forward neural network.

        Return:
            Prediction object with sentence scores.
        """
        return Prediction(score=self.estimator(sentemb).view(-1))

    def compute_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_encoder_last_hidden_states: bool = False,
    ) -> torch.Tensor:
        """Function that extracts sentence embeddings for
        a single sentence.

        Args:
            input_ids (torch.Tensor): sequences [batch_size x seq_len].
            attention_mask (torch.Tensor): attention_mask [batch_size x seq_len].
            return_encoder_last_hidden_states (bool): If set to True, returns the last encoder last hidden states.

        Returns:
            torch.Tensor [batch_size x hidden_size] with sentence embeddings.
        """
        return self.encoder(
            input_ids,
            attention_mask,
            return_last_hidden_states=return_encoder_last_hidden_states,
        )

    def get_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_encoder_last_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Function that extracts sentence embeddings for
        a single sentence.

        Args:
            input_ids (torch.Tensor): sequences [batch_size x seq_len].
            attention_mask (torch.Tensor): attention_mask [batch_size x seq_len].
            return_encoder_last_hidden_states (bool): If set to True, returns the last encoder last hidden states.

        Returns:
            torch.Tensor [batch_size x hidden_size] with sentence embeddings.
        """
        return self.compute_sentence_embedding(
            input_ids,
            attention_mask,
            return_encoder_last_hidden_states=return_encoder_last_hidden_states,
        )

    def forward(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        return_encoder_last_hidden_states: bool = False,
        **kwargs,
    ) -> Union[Prediction, Tuple[Prediction, torch.tensor]]:
        """Regression model forward method.

        Args:
            input_ids (torch.tensor): Input ids of the sentences in the batch.
            attention_mask (torch.tensor): Attention mask for the sentences in the batch.
            return_encoder_last_hidden_states (bool): If set to True, returns the last encoder last hidden states.

        Return:
            Union[Prediction, Tuple[Prediction, torch.tensor]]: Prediction object with translation scores, eventually
                                                                with encoder last hidden states.
        """
        sentemb = self.get_sentence_embedding(
            input_ids,
            attention_mask,
            return_encoder_last_hidden_states=return_encoder_last_hidden_states,
        )
        return (
            (self.estimate(sentemb[:, 0, :]), sentemb)
            if return_encoder_last_hidden_states
            else self.estimate(sentemb)
        )

    def read_training_data(self, path: Path) -> List[Dict[str, Union[str, float]]]:
        """Method that reads the training data from a csv file and returns a list of samples.

        Args:
            path (Path): Path to the csv file containing the training data.

        Returns:
            List[Dict[str, Union[str, float]]]: List of dictionaries containing training samples.
        """
        return read_csv_data(
            path, {self.hparams.sent_to_use: "str", "score": "float16"}
        )

    def read_validation_data(self, path: Path) -> List[dict]:
        """Method that reads the validation data from a csv file and returns a list of samples.

        Args:
            path (Path): Path to the csv file containing the validation data.

        Returns:
            List[Dict[str, Union[str, float]]]: List of dictionaries containing validation samples.
        """
        return read_csv_data(
            path, {self.hparams.sent_to_use: "str", "score": "float16"}
        )
