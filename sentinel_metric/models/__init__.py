from pathlib import Path
from typing import Union
import os

import torch
import yaml
from huggingface_hub import snapshot_download

from .base import RegressionMetricModel
from .sentinel_regression.sentinel_regression_metric import SentinelRegressionMetric

str2model = {"sentinel_regression_metric": SentinelRegressionMetric}


def download_model(model: str, saving_directory: Union[str, Path, None] = None, local_files_only: bool = False) -> str:
    """Download an MT Sentinel Metric model from the Hugging Face Hub, returning its cache path.

    Args:
        model (str): Name of the HF model to be loaded.
        saving_directory (Union[str, Path, None]):  Path to the folder where cached files are stored.
        local_files_only (bool): Whether avoid download, returning the path to the local cached file if it exists.

    Returns:
        str: Local path to model checkpoint.
    """
    try:
        model_path = snapshot_download(
            repo_id=model, cache_dir=saving_directory, local_files_only=local_files_only
        )
    except Exception:
        raise KeyError(f"Model '{model}' not found.")
    else:
        checkpoint_path = os.path.join(*[model_path, "checkpoints", "model.ckpt"])
    return checkpoint_path


def load_from_checkpoint(
    checkpoint_path: str,
    reload_hparams: bool = False,
    strict: bool = False,
    class_identifier: str = "sentinel_regression_metric",
) -> RegressionMetricModel:
    """Loads models from a checkpoint path.

    Args:
        checkpoint_path (str): Path to a model checkpoint.
        reload_hparams (bool): hparams.yaml file located in the parent folder is
            only use for deciding the `class_identifier`. By setting this flag
            to True all hparams will be reloaded.
        strict (bool): Strictly enforce that the keys in checkpoint_path match the
            keys returned by this module's state dict. Defaults to False
        class_identifier (str): String that identifies the model class with which load
            the pre-trained weights. This info must be present either in the hparams or
            in this parameter.
    Return:
        RegressionMetricModel model.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise Exception(f"Invalid checkpoint path: {checkpoint_path}")

    parent_folder = checkpoint_path.parents[1]
    hparams_file = parent_folder / "hparams.yaml"
    if hparams_file.is_file():
        with open(hparams_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model_class = str2model[hparams["class_identifier"]]
    elif reload_hparams:
        raise Exception(
            f"Input parameter reload_hparams=True, but hparams.yaml file is missing from {parent_folder}!"
        )
    else:
        model_class = str2model[class_identifier]

    model = model_class.load_from_checkpoint(
        checkpoint_path,
        load_pretrained_weights=False,
        hparams_file=hparams_file if reload_hparams else None,
        map_location=torch.device("cpu"),
        strict=strict,
    )
    return model
