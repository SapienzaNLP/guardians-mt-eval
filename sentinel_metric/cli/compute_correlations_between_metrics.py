import itertools
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set, Callable, Any
from tqdm import tqdm

import scipy.stats
import numpy as np
from mt_metrics_eval import data, stats
from matplotlib import pyplot as plt
import seaborn as sns

from sentinel_metric.cli.compute_correlations_on_wmt import get_metric_name2scores
from sentinel_metric.cli.score import get_wmt_testset

sentinel_metric2latex = {
    "SENTINEL-CAND-MQM": r"$\text{SENTINEL}_{\text{CAND}}$",
    "SENTINEL-SRC-MQM": r"$\text{SENTINEL}_{\text{SRC}}$",
    "SENTINEL-REF-MQM": r"$\text{SENTINEL}_{\text{REF}}$",
}

plt.rcParams["text.usetex"] = False


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to compute the correlations between metrics on the WMT test set passed in input and save "
        "them in a output heatmap file."
    )
    parser.add_argument(
        "--metrics-to-evaluate-info-filepath",
        type=Path,
        required=True,
        help="Path to the file containing the info for metrics to evaluate.",
    )
    parser.add_argument(
        "--testset-name",
        type=str,
        default="wmt23",
        help="Name of the WMT test set to use. Default: 'wmt23'.",
    )
    parser.add_argument(
        "--lp",
        type=str,
        default="zh-en",
        help="Language pair to consider in the test set passed in input. Default: 'zh-en'.",
    )
    parser.add_argument(
        "--ref-to-use",
        type=str,
        default="refA",
        help="Which human reference to use (it will be used iff the metric is ref-based). It must be like refA, refB, "
        "etc. Default: 'refA'.",
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        help="Whether to include 'human' systems (i.e., reference translations) among systems.",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Whether to include systems considered to be outliers.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        required=True,
        help="Path to the file where to save report.",
    )
    return parser


def compute_correlations_between_metrics_command() -> None:
    """Command to compute the correlations between metrics on the WMT test set passed in input."""
    parser = read_arguments()
    args = parser.parse_args()

    metric_name2scores = get_metric_name2scores(
        args.metrics_to_evaluate_info_filepath, args.ref_to_use
    )

    testset = get_wmt_testset(args.testset_name, args.lp, True)

    compute_correlations_between_metrics(
        testset,
        args.include_human,
        args.include_outliers,
        args.ref_to_use,
        metric_name2scores,
        args.out_file,
    )


def get_correlation_value(
    testset: data.EvalSet,
    scores_1: Dict[str, List[float]],
    scores_2: Dict[str, List[float]],
    sys_names: Set[str],
    corr_fcn: Callable,
    **corr_fcn_args: Any,
) -> float:
    """Compute the correlation between two metrics' scores.

    Args:
        testset (data.EvalSet): The WMT test set to use.
        scores_1 (Dict[str, List[float]]): Scores returned by the first metric.
        scores_2 (Dict[str, List[float]]): Scores returned by the second metric.
        sys_names (Set[str]): System names to consider in the correlation.
        corr_fcn (Callable[[List[float], List[float], ...], Tuple[float, float]]): Correlation function to use. It must
                                                                                   be in [scipy.stats.kendalltau,
                                                                                   stats.KendallWithTiesOpt,
                                                                                   scipy.stats.pearsonr].
        **corr_fcn_args (Any): Optional extra arguments to corr_fcn.

    Returns:
        float: The computed correlation value.
    """
    if corr_fcn not in [
        scipy.stats.kendalltau,
        stats.KendallWithTiesOpt,
        scipy.stats.pearsonr,
    ]:
        raise ValueError(
            "Correlation function not allowed for 'get_correlation_value' method. Choose from scipy.stats.kendalltau, "
            "stats.KendallWithTiesOpt, or scipy.stats.pearsonr."
        )

    correlation_obj = testset.Correlation(
        scores_1,
        scores_2,
        sys_names,
    )
    corr_wrapper = stats.AverageCorrelation(
        corr_fcn,
        correlation_obj.num_sys,
        average_by="item" if corr_fcn == stats.KendallWithTiesOpt else "none",
        filter_nones=correlation_obj.none_count,
        replace_nans_with_zeros=False,
        **corr_fcn_args,
    )
    corr_value = corr_wrapper(
        correlation_obj.gold_scores, correlation_obj.metric_scores
    )[0]
    return corr_value


def generate_heatmap_matplotlib(
    metrics: List[str],
    correlations: Dict[Tuple[str, str], float],
    filepath: Path,
) -> None:
    """Generate a heatmap with the correlation matrix between metrics using matplotlib.

    Args:
        metrics (List[str]): List of metrics to consider.
        correlations (Dict[Tuple[str, str], float]): Dictionary with the correlation values between metrics.
        filepath (Path): Path to the file where to save the heatmap.
    """
    correlation_matrix = []
    for metric1 in metrics:
        corr_row = []
        for metric2 in metrics:
            if metric1 == metric2:
                corr = 1.0
            else:
                assert (metric1, metric2) not in correlations or (
                    metric2,
                    metric1,
                ) not in correlations
                corr = correlations.get((metric1, metric2), 0) + correlations.get(
                    (metric2, metric1), 0
                )
            corr_row.append(round(corr, 3))
        correlation_matrix.append(corr_row)
    correlation_matrix = np.array(correlation_matrix)

    mask = np.tril(np.ones(correlation_matrix.shape), k=0)
    correlation_matrix = np.ma.filled(
        np.ma.array(correlation_matrix, mask=mask), fill_value=np.nan
    )

    # Creating an upper triangular matrix
    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))
    correlation_matrix[mask] = np.nan

    fig = plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        correlation_matrix,
        cmap="BuGn",
        annot=True,
        fmt=".3f",
        annot_kws={"size": 10},
        xticklabels=metrics,
        yticklabels=metrics,
    )

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, format="pdf")  # Save the figure


def adjust_metric_name_for_display(metric_name: str) -> str:
    """Adjusts the metric name for display in the heatmap correlation matrix.

    Args:
        metric_name (str): The metric name to adjust.

    Returns:
        str: The adjusted metric name.
    """
    if metric_name.endswith("-src"):
        metric_name = metric_name[:-4]
    elif metric_name.endswith("-refA") or metric_name.endswith("-refB"):
        metric_name = metric_name[:-5]

    if metric_name in sentinel_metric2latex:
        metric_name = sentinel_metric2latex[metric_name]

    return metric_name


def compute_correlations_between_metrics(
    testset: data.EvalSet,
    include_human: bool,
    include_outliers: bool,
    ref_to_use: str,
    metric_name2scores: Dict[
        str, Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]
    ],
    out_file: Path,
) -> None:
    """Compute the correlation between metrics on the WMT test set passed in input and saves them in a file.

    Args:
        testset (data.EvalSet): The WMT test set to use.
        include_human (bool): Whether to include 'human' systems (i.e., reference translations) among systems.
        include_outliers (bool): Whether to include systems considered to be outliers.
        ref_to_use (str): Human reference used in the test set passed in input.
        metric_name2scores (Dict[str, Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]]):
                            Dictionary from metric name to its scores.
        out_file (Path): Path to the file where to save the report.
    """
    for metric_name, (seg_scores, sys_scores) in metric_name2scores.items():
        if metric_name in testset.metric_names:
            metric_name2scores[metric_name] = (
                testset.Scores("seg", metric_name),
                testset.Scores("sys", metric_name),
            )
        elif seg_scores is None or sys_scores is None:
            raise ValueError(
                f"Metric {metric_name}'s outputs not passed in input and not present in WMT."
            )

    sys_names = testset.sys_names - {ref_to_use}
    if not include_human:
        sys_names -= testset.human_sys_names
    if not include_outliers:
        sys_names -= testset.outlier_sys_names

    # Generate all possible pairs of metrics
    metric_pairs = list(itertools.combinations(metric_name2scores, 2))
    correlations = dict()
    for metric_name_1, metric_name_2 in tqdm(
        metric_pairs, desc="Computing correlations"
    ):
        sys_names_to_consider = (
            sys_names
            & set(metric_name2scores[metric_name_1][0])
            & set(metric_name2scores[metric_name_2][0])
        )

        corr_value = get_correlation_value(
            testset,
            metric_name2scores[metric_name_1][0],
            metric_name2scores[metric_name_2][0],
            sys_names_to_consider,
            scipy.stats.pearsonr,
        )

        metric_name_1 = adjust_metric_name_for_display(metric_name_1)
        metric_name_2 = adjust_metric_name_for_display(metric_name_2)

        correlations[(metric_name_1, metric_name_2)] = corr_value

    metric_names = [
        adjust_metric_name_for_display(metric_name)
        for metric_name in metric_name2scores
    ]
    generate_heatmap_matplotlib(metric_names, correlations, out_file)


if __name__ == "__main__":
    compute_correlations_between_metrics_command()
