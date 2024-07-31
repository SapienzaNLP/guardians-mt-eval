import itertools
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set, Callable, Any
from tqdm import tqdm

import scipy.stats
import numpy as np
import plotly.graph_objects as go
from mt_metrics_eval import data, stats
from matplotlib import pyplot as plt
import seaborn as sns

from sentinel_metric.scripts.compute_correlations_on_wmt import get_metric_name2scores
from sentinel_metric.scripts.score import get_wmt_testset

fake_metrics2latex = {
    "CAND-ONLY-FAKE-METRIC-MQM": r"$\textsc{sentinel}_{\textsc{cand}}$",
    "SRC-ONLY-FAKE-METRIC-MQM": r"$\textsc{sentinel}_{\textsc{src}}$",
    "REF-ONLY-FAKE-METRIC-MQM": r"$\textsc{sentinel}_{\textsc{ref}}$",
}

plt.rcParams["text.usetex"] = True


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to write correlations between Machine Translation metrics in an output html file."
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


def get_correlation_value(
    testset: data.EvalSet,
    scores_1: Dict[str, List[float]],
    scores_2: Dict[str, List[float]],
    sys_names: Set[str],
    corr_fcn: Callable,
    macro: bool = True,
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
        macro (bool): Whether to compute correlations with a plain average over row- or item-wise correlations.
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
        macro=macro,
        **corr_fcn_args,
    )
    corr_value = corr_wrapper(
        correlation_obj.gold_scores, correlation_obj.metric_scores
    )[0]
    return corr_value


def generate_html_report(
    metrics: List[str],
    correlations: Dict[Tuple[str, str], float],
    testset: data.EvalSet,
    filepath: Path,
) -> None:
    """Generate an HTML report with correlations colored from red to green.

    Args:
        metrics (List[str]): List of metric names.
        correlations (Dict[Tuple[str, str], float]): Dictionary from metrics pair to their correlation value.
        testset (data.EvalSet): WMT test set to take into account.
        filepath (Path): Path to the file where to save report.
    """

    def correlation_to_color(value: float) -> str:
        """Convert a correlation value to a color, from red (low) to green (high).

        Args:
            value (float): The correlation value to convert.

        Returns:
            str: The color corresponding to the correlation value.
        """
        if value is None:
            return "#FFFFFF"  # White for missing values
        # Normalize value to be between 0 and 1
        normalized_value = (value + 1) / 2
        red = 255 * (1 - normalized_value)
        green = 255 * normalized_value
        color = f"#{int(red):02X}{int(green):02X}00"
        return color

    colors = {
        metric_pair: correlation_to_color(correlations[metric_pair])
        for metric_pair in correlations
    }

    with open(filepath, "w") as f:
        f.write(
            "<html>\n<head>\n<style>td { text-align: center; }</style>\n</head>\n<body>\n"
        )
        f.write("<table border='1'>\n")
        f.write(
            "<tr><th>Metric</th>"
            + "".join(f"<th>{testset.DisplayName(metric)}</th>" for metric in metrics)
            + "</tr>\n"
        )

        for metric1 in metrics:
            f.write(f"<tr><td><b>{testset.DisplayName(metric1)}</b></td>")
            for metric2 in metrics:
                # Attempt to get the color for both possible orders of metric1 and metric2
                color = colors.get(
                    (metric1, metric2), colors.get((metric2, metric1), "#FFFFFF")
                )

                # Similarly, attempt to get the correlation for both orders, defaulting to '' if not found
                correlation = correlations.get(
                    (metric1, metric2), correlations.get((metric2, metric1), "")
                )
                f.write(
                    f"<td bgcolor='{color}'>{round(correlation, 2) if correlation != '' else correlation}</td>"
                )
            f.write("</tr>\n")

        f.write("</table>\n</body>\n</html>\n")


def generate_heatmap_matplotlib(
    metrics: list,
    correlations: dict,
    filepath: Path,
) -> None:
    # Assuming the correlation matrix preparation code remains unchanged

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

    # Adjusting the appearance
    # ax.set_xticks(np.arange(len(metrics)))
    # ax.set_yticks(np.arange(len(metrics)))
    # ax.set_xticklabels(metrics)
    # ax.set_yticklabels(metrics)

    # plt.xticks(rotation=90)
    # ax.xaxis.set_ticks_position("top")
    # ax.xaxis.set_label_position("top")

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, format="pdf")  # Save the figure


def generate_heatmap_plotly(
    metrics: List[str],
    correlations: Dict[Tuple[str, str], float],
    filepath: Path,
) -> None:
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
    correlation_matrix = np.flipud(correlation_matrix)

    annotations = []
    for i, row in enumerate(correlation_matrix):
        for j, value in enumerate(row):
            if not np.isnan(value):  # Only add annotation if value is not NaN
                annotations.append(
                    dict(
                        y=metrics[len(metrics) - 1 - i],
                        x=metrics[j],
                        text="{:.3f}".format(value),
                        showarrow=False,
                    )
                )

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            x=metrics,
            y=list(reversed(metrics)),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            showscale=True,
            xgap=1,
            ygap=1,
        )
    )

    # Update the layout to make it more readable and adjust the color scale
    fig.update_layout(
        # title="Upper Triangular Correlation Matrix",
        annotations=annotations,
    )
    fig.update_xaxes(side="top")
    fig["data"][0]["showscale"] = True
    fig.update_traces(showscale=True, colorbar=dict(tickvals=[-1, -0.5, 0, 0.5, 1]))
    fig.update_layout(
        xaxis=dict(tickangle=90, automargin=True),
        yaxis=dict(automargin=True),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
    )
    # Show the figure
    fig.write_image(filepath, width=1200, height=1000, scale=2)


def adjust_metric_name_for_display(metric_name: str):
    if metric_name.endswith("-src"):
        metric_name = metric_name[:-4]
    elif metric_name.endswith("-refA") or metric_name.endswith("-refB"):
        metric_name = metric_name[:-5]

    if metric_name in fake_metrics2latex:
        metric_name = fake_metrics2latex[metric_name]

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
    """Compute the correlation between metrics on the WMT test set passed in input and saves them in a html file.

    Args:
        testset (data.EvalSet): The WMT test set to use.
        include_human (bool): Whether to include 'human' systems (i.e., reference translations) among systems.
        include_outliers (bool): Whether to include systems considered to be outliers.
        ref_to_use (str): Human reference used in the test set passed in input.
        metric_name2scores (Dict[str, Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]]):
                            Dictionary from metric name to its scores.
        out_file (Path): Path to the file where to save report.
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

    # generate_html_report(list(metric_name2scores), correlations, testset, out_file)
    metric_names = [
        adjust_metric_name_for_display(metric_name)
        for metric_name in metric_name2scores
    ]
    generate_heatmap_matplotlib(metric_names, correlations, out_file)


if __name__ == "__main__":
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
