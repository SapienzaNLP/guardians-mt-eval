r"""
MTMetric
========================
    Abstract Metric class that provides all base logic for MT metrics.
    Extend this class to create a new MT metric.
"""

from abc import ABC, abstractmethod

from typing import Dict, List

from sentinel_metric.models.utils import Prediction


class MTMetric(ABC):
    def __init__(self, name: str, **kwargs):
        """Metric for MT evaluation.

        Args:
            name (str): The name of the metric.
        """
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def predict(self, samples: List[Dict[str, str]]) -> Prediction:
        """Method that receives a list of samples (dictionaries with translations,
        sources and/or references) and returns segment-level scores and system level score.

        Args:
            samples (List[Dict[str, str]]): List with dictionaries with source,
                translations and/or references.

        Return:
            Prediction object with `scores` and `system_score`
        """
        pass
