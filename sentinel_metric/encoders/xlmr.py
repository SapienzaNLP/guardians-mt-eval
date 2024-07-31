r"""
XLM-RoBERTa Encoder
==============
    Pretrained XLM-RoBERTa  encoder from Hugging Face.
"""
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizerFast


class XLMREncoder(nn.Module):
    """XLM-RoBERTA encoder.

    Args:
        pretrained_model (str): Pretrained model from hugging face.
        load_pretrained_weights (bool): If set to True loads the pretrained weights
            from Hugging Face
    """

    def __init__(
        self, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> None:
        super().__init__()
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(pretrained_model)
        if load_pretrained_weights:
            self.model = XLMRobertaModel.from_pretrained(
                pretrained_model, add_pooling_layer=False
            )
        else:
            self.model = XLMRobertaModel(
                XLMRobertaConfig.from_pretrained(pretrained_model),
                add_pooling_layer=False,
            )
        self.model.encoder.output_hidden_states = False

    @property
    def output_units(self) -> int:
        """Hidden size of the encoder model."""
        return self.model.config.hidden_size

    def prepare_sample(
        self,
        sample: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Receives a list of strings and applies tokenization and vectorization.

        Args:
            sample (List[str]): List with text segments to be tokenized and padded.

        Returns:
            Dict[str, torch.Tensor]: dict with 'input_ids' and 'attention_mask'
        """
        tokenizer_output = self.tokenizer(
            sample,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return tokenizer_output

    @classmethod
    def from_pretrained(
        cls, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> "XLMREncoder":
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str):Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face

        Returns:
            Encoder: XLMREncoder object.
        """
        return XLMREncoder(pretrained_model, load_pretrained_weights)

    def freeze_embeddings(self) -> None:
        """Frezees the embedding layer."""
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_last_hidden_states=False,
    ) -> torch.Tensor:
        """Forward pass of the encoder.

        Args:
            input_ids (torch.Tensor): Tensor of input ids.
            attention_mask (torch.Tensor): Tensor of attention masks.
            return_last_hidden_states (bool): Flag to return all last hidden states. Default is False.
        Returns:
            torch.Tensor: Last hidden states tensor. If return_last_hidden_states is True, returns all last hidden
                          states. Otherwise, returns the last hidden state corresponding to the `[CLS]` token.
        """
        last_hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

        # If return_last_hidden_states is True, return all hidden states
        if return_last_hidden_states:
            return last_hidden_states
        else:
            # Return only the last hidden state corresponding to the `[CLS]` token
            return last_hidden_states[:, 0, :]
