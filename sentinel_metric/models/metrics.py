r"""
Metrics
=======
    Metrics to be used during training to measure correlations with human judgements
"""
from typing import Any, Callable, List, Optional, Dict

import scipy.stats as stats
import torch
from torchmetrics import Metric


class RegressionMetrics(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    preds: List[torch.Tensor]
    target: List[torch.Tensor]

    def __init__(
        self,
        prefix: str = "",
        compute_correlations: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.compute_correlations = compute_correlations
        if compute_correlations:
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("mse_losses", default=[], dist_reduce_fx="cat")
        self.prefix = prefix

    def update(
        self,
        preds: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        mse_loss: Optional[torch.Tensor] = None,
    ) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds (Optional[torch.Tensor]): Predictions from model.
            target (Optional[torch.Tensor]): Ground truth values.
            mse_loss (Optional[torch.Tensor]): MSE loss tensor.
        """
        if self.compute_correlations:
            self.preds.append(preds)
            self.target.append(target)

        if mse_loss is not None:
            self.mse_losses.append(mse_loss.detach())

    def compute(self) -> Dict[str, float]:
        """Computes metrics."""
        kendall = None
        if self.compute_correlations:
            try:
                preds = torch.cat(self.preds, dim=0)
                target = torch.cat(self.target, dim=0)
            except TypeError:
                preds = self.preds
                target = self.target
            kendall, _ = stats.kendalltau(preds.tolist(), target.tolist())

        mse_loss_avg = (
            torch.mean(torch.stack(self.mse_losses))
            if self.mse_losses
            else torch.tensor(0.0)
        )

        report = {self.prefix + "_mse_loss": mse_loss_avg.item()}
        if self.compute_correlations:
            report[self.prefix + "_kendall_correlation"] = kendall
        return report
