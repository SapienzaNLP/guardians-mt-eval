# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An implementation of the tau optimization procedure.

See https://arxiv.org/abs/2305.14324 for more details on the optimization
routine.
"""

import dataclasses
import statistics
from typing import Callable, Tuple, Set, List, Optional

import numpy as np
import numpy.typing
import random


class TauSufficientStats:
    """Represents the sufficient statistics for calculating Kendall's tau.

    The two vectors of scores that are correlated are assumed to represent
    human and metric scores. Some taus are asymmetric, so we keep the semantics
    of the vectors to avoid confusion, which could happen if generic names
    were used. If you are calculating the correlation between two metrics, make
    sure to understand whether the tau you are computing is symmetric or not.

    Attributes:
      con: The number of concordant pairs.
      dis: The number of discordant pairs.
      ties_human: The number of pairs tied only in the human scores.
      ties_metric: The number of pairs tied only in the metric scores.
      ties_both: The number of pairs tied in both the human and metric scores.
      num_pairs: The total number of pairs.
    """

    def __init__(
        self,
        con: int = 0,
        dis: int = 0,
        ties_human: int = 0,
        ties_metric: int = 0,
        ties_both: int = 0,
    ):
        self.con = con
        self.dis = dis
        self.ties_human = ties_human
        self.ties_metric = ties_metric
        self.ties_both = ties_both
        self.num_pairs = con + dis + ties_human + ties_metric + ties_both

    def tau_23(self) -> float:
        return (
            self.con + self.ties_both - self.dis - self.ties_human - self.ties_metric
        ) / self.num_pairs

    def acc_23(self) -> float:
        return (self.con + self.ties_both) / self.num_pairs

    def __eq__(self, other: "TauSufficientStats") -> bool:
        return (
            self.con,
            self.dis,
            self.ties_human,
            self.ties_metric,
            self.ties_both,
        ) == (
            other.con,
            other.dis,
            other.ties_human,
            other.ties_metric,
            other.ties_both,
        )

    def __iadd__(self, other: "TauSufficientStats") -> "TauSufficientStats":
        self.con += other.con
        self.dis += other.dis
        self.ties_human += other.ties_human
        self.ties_metric += other.ties_metric
        self.ties_both += other.ties_both
        self.num_pairs += other.num_pairs
        return self

    def __isub__(self, other: "TauSufficientStats") -> "TauSufficientStats":
        self.con -= other.con
        self.dis -= other.dis
        self.ties_human -= other.ties_human
        self.ties_metric -= other.ties_metric
        self.ties_both -= other.ties_both
        self.num_pairs -= other.num_pairs
        return self

    def __str__(self) -> str:
        return (
            "("
            + ",".join(
                [
                    f"C={self.con}",
                    f"D={self.dis}",
                    f"T_h={self.ties_human}",
                    f"T_m={self.ties_metric}",
                    f"T_hm={self.ties_both}",
                ]
            )
            + ")"
        )

    def __repr__(self):
        return str(self)


@dataclasses.dataclass
class TauOptimizationResult:
    thresholds: List[float]
    taus: List[float]
    best_threshold: Tuple[
        float, float
    ]  # For the Tuple types, the first elem is the avg and the second is the std_dev.
    best_tau: Tuple[float, float]
    best_metric_ties: Tuple[float, float]
    n_dev_rows: Tuple[float, float]
    n_test_rows: Tuple[float, float]


class _RankedPair:
    """Maintains the metadata for a ranked pair for calculating Kendall's tau.

    Attributes:
      row: The index of the row in the N x M matrix of scores that this pair
        belongs to.
      diff: The absolute difference between metric scores.
      stats: The tau sufficient statistics that this pair represents.
      tie_stats: The tau sufficient statistics that this pair represents when a
        tie is introduced in the metric score.
    """

    def __init__(self, h1: float, h2: float, m1: float, m2: float, row: int):
        self.row = row
        self.diff = abs(m1 - m2)

        # Determine the sufficient stats for the example when treated normally.
        if h1 == h2 and m1 == m2:
            self.stats = TauSufficientStats(ties_both=1)
        elif h1 == h2:
            self.stats = TauSufficientStats(ties_human=1)
        elif m1 == m2:
            self.stats = TauSufficientStats(ties_metric=1)
        elif (h1 > h2 and m1 > m2) or (h1 < h2 and m1 < m2):
            self.stats = TauSufficientStats(con=1)
        else:
            self.stats = TauSufficientStats(dis=1)

        # Determine the sufficient stats for the example when a tie is introduced
        # in the metric score.
        if h1 == h2:
            self.tie_stats = TauSufficientStats(ties_both=1)
        else:
            self.tie_stats = TauSufficientStats(ties_metric=1)


def _enumerate_pairs(
    human_scores: np.ndarray,
    metric_scores: np.ndarray,
    pairwise_comparisons: Optional[Set[Tuple[int, int, int]]],
    sample_rate: float,
    held_out_rows: Set[int] = None,
    filter_nones: bool = True,
) -> Tuple[list[_RankedPair], set[int]]:
    """Enumerates pairs for Kendall's tau."""
    mat1 = human_scores
    mat2 = metric_scores
    pairs = []
    rows = set()
    for row, (r1, r2) in enumerate(zip(mat1, mat2)):
        if held_out_rows is not None and row in held_out_rows:
            continue

        # Filter Nones
        if filter_nones and (pairwise_comparisons is None or held_out_rows is not None):
            filt = [
                (v1, v2) for v1, v2 in zip(r1, r2) if v1 is not None and v2 is not None
            ]
            if not filt:
                continue
            r1, r2 = zip(*filt)

        for i in range(len(r1)):
            for j in range(i + 1, len(r1)):
                if (
                    pairwise_comparisons is not None
                    and (i, j, row) not in pairwise_comparisons
                ):
                    continue
                if sample_rate == 1.0 or np.random.random() <= sample_rate:
                    if any(score is None for score in [r1[i], r1[j], r2[i], r2[j]]):
                        raise ValueError(
                            f"None scores found in row {row} for pair {i},{j}. The scores are {r1[i]}, {r1[j]}, "
                            f"{r2[i]}, {r2[j]} (human_score_i, human_score_j, metric_score_i, metric_score_j)."
                        )
                    pairs.append(_RankedPair(r1[i], r1[j], r2[i], r2[j], row))
                    rows.add(row)

    return pairs, rows


def tau_optimization(
    metric_scores: numpy.typing.ArrayLike,
    human_scores: numpy.typing.ArrayLike,
    tau_fn: Callable[[TauSufficientStats], float],
    pairwise_comparisons_list: Optional[List[Set[Tuple[int, int, int]]]] = None,
    use_sampled_pairs_only_to_compute_eps: bool = False,
    use_held_out: bool = False,
    sample_rate: float = 1.0,
) -> TauOptimizationResult:
    """Runs tau optimization on the metric scores.

    Tau optimization automatically introduces ties into the metric scores to
    optimize a tau function. For more details, see
    https://arxiv.org/abs/2305.14324.

    The tau value that is calculated and optimized for is the average correlation
    (defined by tau_fn) calculated over paired rows in `metric_scores` and
    `human_scores`.

    If either `metric_scores` or `human_scores` are missing values, the
    corresponding entries should be `None`. In such cases, the input type should
    be a Python list or a NumPy array with dtype=object. If `np.nan` is used
    instead, the missing values will not be properly removed.

    Args:
      metric_scores: An N x M matrix of metric scores.
      human_scores: An N x M matrix of human scores.
      tau_fn: The tau function to optimize for. This can be a function like
        `TauSufficientStats.acc_23`
      pairwise_comparisons_list: Which pairwise comparisons to consider.
      use_sampled_pairs_only_to_compute_eps: Whether to use the sampled pairs only
                                             to compute the optimal threshold (as a
                                             sort of alternative to the 'sample_rate'
                                             param).
      use_held_out: Whether to use the pairs in `pairwise_comparisons_list` as held out where to compute the thresholds.
      sample_rate: The proportion of all possible pairs to consider when searching
        for epsilon and calculating tau. Must be in the range (0, 1]. Any value
        less than 1 will mean the search and optimal tau will be approximations.
        The sampling is random and uses `np.random`, so it can be made
        deterministic by fixing the NumPy random seed.

    Returns:
      The optimization result.
    """
    if use_held_out:
        use_sampled_pairs_only_to_compute_eps = False

    if use_sampled_pairs_only_to_compute_eps or use_held_out:
        sample_rate = 1

    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError(
            f"`sample_rate` must be in the range (0, 1]. Found {sample_rate}"
        )

    # Convert the data to a numpy array in case it isn't already.
    metric_scores = np.array(metric_scores)
    human_scores = np.array(human_scores)

    # The optimization routine expects 2-dimensional matrices. If we are only
    # given vectors, create a dummy dimension.
    if metric_scores.ndim == 1:
        metric_scores = np.expand_dims(metric_scores, 0)
    if human_scores.ndim == 1:
        human_scores = np.expand_dims(human_scores, 0)

    if human_scores.shape != metric_scores.shape:
        raise ValueError("Human and metric scores must have the same shape.")

    thresholds_list, taus_list, max_threshold_list, max_tau_list, metric_ties_list = (
        [],
        [],
        [],
        [],
        [],
    )
    n_dev_rows_list, n_test_rows_list = [], []

    if pairwise_comparisons_list is None:
        pairwise_comparisons_list = [None]

    for pairwise_comparisons in pairwise_comparisons_list:
        pairs, rows = _enumerate_pairs(
            human_scores,
            metric_scores,
            pairwise_comparisons,
            sample_rate,
        )
        num_rows = len(rows)
        n_dev_rows_list.append(num_rows)

        # Initialize the sufficient stats per row
        row_to_stats = {row: TauSufficientStats() for row in rows}
        metric_ties = [0]
        for pair in pairs:
            row_to_stats[pair.row] += pair.stats
            metric_ties[0] += pair.stats.ties_metric + pair.stats.ties_both

        # Initialize the optimization. We start with a threshold of 0.0, representing
        # no new ties introduced. This is necessary in case there are no ties in
        # the metric score at all (meaning epsilon=0 will not be a candidate) and
        # introducing any ties is bad.
        thresholds = [0.0]
        total_tau = sum(tau_fn(stats) for stats in row_to_stats.values())
        taus = [total_tau / num_rows]

        # Search all pairs to find the best tau value.
        pairs.sort(key=lambda p: p.diff)
        for pair in pairs:
            # Remove the old tau from the overall sum
            total_tau -= tau_fn(row_to_stats[pair.row])

            # Remove this pair from the overall counts, then reintroduce it as a tie.
            row_to_stats[pair.row] -= pair.stats
            row_to_stats[pair.row] += pair.tie_stats

            # Add the tau back to the overall average
            total_tau += tau_fn(row_to_stats[pair.row])

            # Save the new overall for this threshold. If we have already calculated
            # a tau for this threshold, overwrite the previous one because each
            # threshold should flip every pair with the equivalent diff and the
            # previous one did not include this tie.
            overall_tau = total_tau / num_rows
            if thresholds[-1] == pair.diff:
                taus[-1] = overall_tau
                metric_ties[-1] += pair.diff != 0
            else:
                thresholds.append(pair.diff)
                taus.append(overall_tau)
                metric_ties.append(metric_ties[-1] + 1)

        # Identify the best eps
        max_index = np.nanargmax(taus)
        max_threshold = thresholds[max_index]

        if use_sampled_pairs_only_to_compute_eps or use_held_out:
            pairs, rows = _enumerate_pairs(
                human_scores,
                metric_scores,
                None,
                sample_rate,
                rows if use_held_out else None,
            )
            num_rows = len(rows)
            n_test_rows_list.append(num_rows)

            # Initialize the sufficient stats per row
            row_to_stats = {row: TauSufficientStats() for row in rows}
            max_metric_ties = 0
            for pair in pairs:
                pair_stats = pair.stats if pair.diff > max_threshold else pair.tie_stats
                row_to_stats[pair.row] += pair_stats
                max_metric_ties += pair_stats.ties_metric + pair_stats.ties_both

            # Compute tau considering the best threshold obtained from the subset
            total_tau = sum(tau_fn(stats) for stats in row_to_stats.values())
            max_tau = total_tau / num_rows
        else:
            n_test_rows_list.append(num_rows)

            max_tau = taus[max_index]
            max_metric_ties = metric_ties[max_index]

        thresholds_list.append(thresholds)
        taus_list.append(taus)
        max_threshold_list.append(max_threshold)
        max_tau_list.append(max_tau)
        metric_ties_list.append(max_metric_ties / len(pairs))

    thresholds = [sum(group) / len(group) for group in zip(*thresholds_list)]
    taus = [sum(group) / len(group) for group in zip(*taus_list)]
    max_threshold_mean, max_threshold_std_dev = (
        statistics.mean(max_threshold_list),
        statistics.stdev(max_threshold_list) if len(max_threshold_list) > 1 else 0,
    )
    max_tau_mean, max_tau_std_dev = (
        statistics.mean(max_tau_list),
        statistics.stdev(max_tau_list) if len(max_tau_list) > 1 else 0,
    )
    max_metric_ties_mean, max_metric_ties_std_dev = (
        statistics.mean(metric_ties_list),
        statistics.stdev(metric_ties_list) if len(metric_ties_list) > 1 else 0,
    )
    n_dev_rows_mean, n_dev_rows_std_dev = (
        statistics.mean(n_dev_rows_list),
        statistics.stdev(n_dev_rows_list) if len(n_dev_rows_list) > 1 else 0,
    )
    n_test_rows_mean, n_test_rows_std_dev = (
        statistics.mean(n_test_rows_list),
        statistics.stdev(n_test_rows_list) if len(n_test_rows_list) > 1 else 0,
    )

    return TauOptimizationResult(
        thresholds,
        taus,
        (max_threshold_mean, max_threshold_std_dev),
        (max_tau_mean, max_tau_std_dev),
        (max_metric_ties_mean, max_metric_ties_std_dev),
        (n_dev_rows_mean, n_dev_rows_std_dev),
        (n_test_rows_mean, n_test_rows_std_dev),
    )
