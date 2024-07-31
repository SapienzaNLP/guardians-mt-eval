r"""
Feed Forward
============
    Feed Forward Neural Network module that can be used for classification or regression
"""
from typing import List, Optional

import torch
from torch import nn


class FeedForward(nn.Module):
    """Feed Forward Neural Network.

    Args:
        in_dim (int): Number input features.
        out_dim (int): Number of output features. Default is just a score.
        hidden_sizes (List[int]): List with hidden layer sizes. Defaults to None.
        activations (str): Name of the activation function to be used in the hidden
            layers. Defaults to 'Tanh'.
        final_activation (Optional[str]): Final activation if any.
        dropout (float): dropout to be used in the hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_sizes: List[int] = None,
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules = [
            nn.Linear(in_dim, hidden_sizes[0]),
            self.build_activation(activations),
            nn.Dropout(dropout),
        ]

        for i in range(1, len(hidden_sizes)):
            modules.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            modules.append(self.build_activation(activations))
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(hidden_sizes[-1], int(out_dim)))
        if final_activation is not None:
            modules.append(self.build_activation(final_activation))

        self.ff = nn.Sequential(*modules)

    def build_activation(self, activation: str) -> nn.Module:
        """Returns the torch activation function whose name matches the one in input.

        Args:
            activation: Activation function name

        Returns:
            torch activation function
        """
        if hasattr(nn, activation.title()):
            return getattr(nn, activation.title())()
        else:
            raise Exception(f"{activation} is not a valid activation function!")

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the feed forward network.

        Args:
            in_features: Input of the feed forward network.

        Returns:
            Output torch tensor of the feed forward network.
        """
        return self.ff(in_features)
