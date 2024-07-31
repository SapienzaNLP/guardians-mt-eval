import json
import pickle
from argparse import ArgumentParser
from pprint import pp
from typing import Optional, List, Dict, Tuple, Union, Set
from pathlib import Path
import random

from mt_metrics_eval import data, stats
import numpy as np
from matplotlib import pyplot as plt

from sentinel_metric.scripts.compute_correlations_on_wmt import get_metric_name2scores
from sentinel_metric.scripts.score import get_wmt_testset


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to compute several KendallWithTiesOpt accuracies varying the test set ties distribution."
    )
    parser.add_argument(
        "--metrics-to-evaluate-info-filepath",
        type=Path,
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
        "--gold-name",
        type=str,
        default="mqm",
        help="Which human ratings to use as gold scores. Default: 'mqm'.",
    )
    parser.add_argument(
        "--primary-metrics",
        action="store_true",
        help="Whether to compare only metrics that have been designated as primary submissions.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Sample rate to pass to tau_optimization for 'KendallWithTiesOpt' (used only in wmt23). Default: 1.0.",
    )
    parser.add_argument(
        "--macro",
        action="store_true",
        help="Whether to compute correlations with a plain average over row- or item-wise correlations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to use for the gold ties and non-ties filtering. Default: 42.",
    )
    parser.add_argument(
        "--n-resampling",
        type=int,
        default=5,
        help="Number of resampling to perform for each iteration. The corresponding acc_eq will be the avg across the "
        "several resampling. Default: 5.",
    )
    parser.add_argument(
        "--use-sampled-pairs-only-to-compute-eps",
        action="store_true",
        help="Whether to use the sampled pairs only to compute the optimal thresholds (as a sort of alternative to the "
        "'sample_rate' param, that will be ignored).",
    )
    parser.add_argument(
        "--held-out-proportion",
        type=float,
        default=0,
        help="If > 0, a held-out set having as proportion wrt the overall test set the input value will be used "
        "to compute the optimal thresholds. The 'sample-rate' and 'use-sampled-pairs-only-to-compute-eps' params will "
        "be ignored. Default: 0.",
    )
    parser.add_argument(
        "--return-std-dev",
        action="store_true",
        help="Whether to return the standard deviation of statistics when performing re-sampling.",
    )
    parser.add_argument(
        "--return-n-rows",
        action="store_true",
        help="Whether to return the number of rows in dev and test sets for KendallWithTiesOpt.",
    )
    parser.add_argument(
        "--ranks-dir-path",
        type=Path,
        required=True,
        help="Path to the directory where the several KendallWithTiesOpt rankings will be saved.",
    )
    parser.add_argument(
        "--plots-dir-path",
        type=Path,
        required=True,
        help="Path to the directory where the script will save the plots containing the several KendallWithTiesOpt "
        "values.",
    )
    parser.add_argument(
        "--grey-continuous-metrics",
        action="store_true",
        help="Whether to use plot in grey all continuous metrics.",
    )
    return parser


def compute_accuracies_varying_ties(
    testset: data.EvalSet,
    ref_to_use: str,
    include_human: bool,
    include_outliers: bool,
    gold_name: str,
    primary_metrics: bool,
    sample_rate: float,
    macro: bool,
    seed: int,
    n_resampling: int,
    use_sampled_pairs_only_to_compute_eps: bool,
    held_out_proportion: float,
    return_std_dev: bool,
    return_n_rows: bool,
    ranks_dir_path: Path,
    metric_name2scores: Dict[
        str, Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]
    ],
) -> List[Dict[str, Union[Tuple[float, float], Dict[str, Tuple[float, float]], float]]]:
    """Compute kendall with ties with different probability of filtering tied and non-tied pairs.

    Args:
        testset (data.EvalSet): The WMT test set to use.
        ref_to_use (str): Which human reference to use.
        include_human (bool): Whether to include 'human' systems (i.e., reference translations) among systems.
        include_outliers (bool): Whether to include systems considered to be outliers.
        gold_name (str): Which human ratings to use as gold scores.
        primary_metrics (bool): Whether to compare only metrics that have been designated as primary submissions.
        sample_rate (float): Sample rate to pass to tau_optimization for 'KendallWithTiesOpt' (used only in wmt23).
        macro (bool): Whether to compute correlations with a plain average over row- or item-wise correlations.
        seed (int): Seed to use for the gold ties and non-ties filtering.
        n_resampling (int): Number of resampling to perform for each iteration. The corresponding acc_eq will be the avg
                            across the several resampling.
        use_sampled_pairs_only_to_compute_eps (bool): Whether to use the sampled pairs only to compute the optimal
                                                      threshold (as a sort of alternative to the 'sample_rate' param).
        held_out_proportion (float): If > 0, it defines the proportion of the test set that will be used as held-out.
        return_std_dev: Whether to return the standard deviation of statistics when performing re-sampling.
        return_n_rows: Whether to return the number of rows in dev and test sets for KendallWithTiesOpt.
        ranks_dir_path (Path): Path to the directory where the several KendallWithTiesOpt rankings will be saved.
        metric_name2scores (Dict[str, Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]]):
                            Dictionary from metric name to its scores.

    Returns:
        List[Dict[str, Union[Tuple[float, float], Dict[str, float], float]]]: List of dictionaries containing the
        KendallWithTiesOpt results for different probability of filtering human ties and non-ties pairs.
    """
    seg_extras, sys_extras = dict(), dict()
    for metric_name, (seg_scores, sys_scores) in metric_name2scores.items():
        if seg_scores is not None:
            seg_extras[metric_name] = seg_scores
            sys_extras[metric_name] = sys_scores

    corrs = data.GetCorrelations(
        testset,
        "seg",
        {ref_to_use},
        set(),
        include_human,
        include_outliers,
        gold_name,
        primary_metrics,
        None,
        extern_metrics=seg_extras,
    )

    held_out_row_ids, test_set_human_ties_proportion = None, None

    def compute_kendall_with_ties(
        metric_corrs: Dict[str, stats.Correlation],
        average_by: str,
        p_filter_gold_ties: float,
        p_filter_gold_non_ties: float,
        curr_held_out_row_ids: Optional[Set[int]] = None,
    ) -> Tuple[
        Dict[str, Tuple[float, float]],
        Dict[str, Union[float, Tuple[float, float]]],
        Union[float, Tuple[float, float]],
        Union[float, Tuple[float, float]],
        Dict[str, Union[float, Tuple[float, float]]],
        Optional[Union[float, Tuple[float, float]]],
        Optional[Union[float, Tuple[float, float]]],
        Optional[Set[int]],
        Optional[float],
        Optional[Dict[str, Tuple[float, float]]],
    ]:
        metrics_comparison_tuple_result = data.CompareMetrics(
            metric_corrs,
            stats.KendallWithTiesOpt,
            average_by,
            macro,
            k=0,
            perm_test="pairs",
            return_metric_name2best_tie_threshold=True,
            return_human_ties_proportion=True,
            p_filter_gold_ties_in_kendall_with_ties=p_filter_gold_ties,
            p_filter_gold_non_ties_in_kendall_with_ties=p_filter_gold_non_ties,
            n_resampling=n_resampling,
            use_sampled_pairs_only_to_compute_eps=use_sampled_pairs_only_to_compute_eps,
            held_out_proportion=held_out_proportion,
            held_out_row_ids=curr_held_out_row_ids,
            return_std_dev=return_std_dev,
            return_n_rows=return_n_rows,
            sample_rate=sample_rate,
        )

        (
            _corrs_and_ranks,
            sig_matrix,
            draws_index,
            draws_list,
            _metric_name2best_tie_threshold,
            _human_ties_proportion,
            _npairs,
            _metric_name2n_ties,
        ) = metrics_comparison_tuple_result[:8]

        curr_n_dev_rows, curr_n_test_rows = (
            metrics_comparison_tuple_result[8:10] if return_n_rows else (None, None)
        )

        curr_test_set_human_ties_proportion = None
        if held_out_proportion > 0 and curr_held_out_row_ids is None:
            if return_n_rows:
                (
                    curr_held_out_row_ids,
                    curr_test_set_human_ties_proportion,
                ) = metrics_comparison_tuple_result[10:12]
            else:
                (
                    curr_held_out_row_ids,
                    curr_test_set_human_ties_proportion,
                ) = metrics_comparison_tuple_result[8:10]
            assert (
                len(curr_held_out_row_ids) > 0
                and curr_test_set_human_ties_proportion > 0
            )

        metric_name2tau_with_std_dev = (
            metrics_comparison_tuple_result[-1] if return_std_dev else None
        )

        return (
            _corrs_and_ranks,
            _metric_name2best_tie_threshold,
            _human_ties_proportion,
            _npairs,
            _metric_name2n_ties,
            curr_n_dev_rows,
            curr_n_test_rows,
            curr_held_out_row_ids,
            curr_test_set_human_ties_proportion,
            metric_name2tau_with_std_dev,
        )

    def load_or_compute(
        _filepath: Path,
        compute_args: Tuple[
            Dict[str, stats.Correlation],
            str,
            float,
            float,
            Optional[Set[int]],
        ],
        prev_test_set_human_ties_proportion: Optional[float] = None,
    ) -> Tuple[
        Dict[
            str,
            Union[
                Tuple[float, float], Dict[str, Union[float, Tuple[float, float]]], float
            ],
        ],
        Optional[Set[int]],
        Optional[float],
    ]:
        if not _filepath.exists():
            (
                corrs_and_ranks,
                metric_name2best_tie_threshold,
                human_ties_proportion,
                n_pairs,
                metric_name2n_ties,
                curr_n_dev_rows,
                curr_n_test_rows,
                curr_held_out_row_ids,
                curr_test_set_human_ties_proportion,
                metric_name2tau_with_std_dev,
            ) = compute_kendall_with_ties(*compute_args)

            result = corrs_and_ranks
            result["metric_name2best_tie_threshold"] = metric_name2best_tie_threshold
            result["human_ties_proportion"] = human_ties_proportion
            result["n_pairs"] = n_pairs
            result["metric_name2n_ties"] = metric_name2n_ties
            result["n_dev_rows"] = curr_n_dev_rows
            result["n_test_rows"] = curr_n_test_rows
            result["test_set_human_ties_proportion"] = (
                curr_test_set_human_ties_proportion
                if curr_test_set_human_ties_proportion is not None
                else prev_test_set_human_ties_proportion
            )

            with open(_filepath, "wb") as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if curr_held_out_row_ids is None:
                curr_held_out_row_ids = set()
            with open(f"{str(_filepath)[:-7]}_held_out_row_ids.pickle", "wb") as handle:
                pickle.dump(
                    curr_held_out_row_ids, handle, protocol=pickle.HIGHEST_PROTOCOL
                )

            if metric_name2tau_with_std_dev is not None:
                tau_with_std_dev_dir = (
                    _filepath.parent / "tau_with_std_dev" / testset.lp
                )
                if not tau_with_std_dev_dir.exists():
                    tau_with_std_dev_dir.mkdir(parents=True, exist_ok=True)
                with open(
                    tau_with_std_dev_dir / f"{_filepath.stem}_tau_with_std_dev.json",
                    "w",
                ) as json_file:
                    json.dump(metric_name2tau_with_std_dev, json_file, indent=4)

        else:
            with open(_filepath, "rb") as handle:
                result = pickle.load(handle)
            with open(f"{str(_filepath)[:-7]}_held_out_row_ids.pickle", "rb") as handle:
                curr_held_out_row_ids = pickle.load(handle)

        return (
            result,
            curr_held_out_row_ids if len(curr_held_out_row_ids) > 0 else None,
            result["test_set_human_ties_proportion"],
        )

    results = []
    random.seed(seed)
    np.random.seed(seed)
    if testset.lp == "he-en":
        list_p_filter_gold_ties = [1.0, 0.9, 0.8, 0.65, 0.5, 0.35, 0.2]
        list_p_filter_non_gold_ties = [0.0, 0.2, 0.4, 0.55, 0.65, 0.75, 1.0]
    else:
        list_p_filter_gold_ties = [1.0, 0.65, 0.30]
        list_p_filter_non_gold_ties = [
            0.0,
            0.2,
            0.4,
            0.5,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            1.0,
        ]

    for p_filter_gold_ties in list_p_filter_gold_ties:
        filepath = (
            ranks_dir_path
            / f"KendallWithTies.result.{p_filter_gold_ties}-0.0.{testset.lp}.pickle"
        )
        (
            exp_results_dict,
            held_out_row_ids,
            test_set_human_ties_proportion,
        ) = load_or_compute(
            filepath,
            (
                corrs,
                "item",
                p_filter_gold_ties,
                0,
                held_out_row_ids,
            ),
            test_set_human_ties_proportion,
        )
        results.append(exp_results_dict)

    for p_filter_gold_non_ties in list_p_filter_non_gold_ties:
        filepath = (
            ranks_dir_path
            / f"KendallWithTies.result.0.0-{p_filter_gold_non_ties}.{testset.lp}.pickle"
        )
        (
            exp_results_dict,
            held_out_row_ids,
            test_set_human_ties_proportion,
        ) = load_or_compute(
            filepath,
            (
                corrs,
                "item",
                0,
                p_filter_gold_non_ties,
                held_out_row_ids,
            ),
            test_set_human_ties_proportion,
        )
        results.append(exp_results_dict)

    return results


def plot_accuracies(
    results: List[
        Dict[str, Union[Tuple[float, float], Dict[str, Tuple[float, float]], float]]
    ],
    plots_dir_path: Path,
    lp: str,
    grey_continuous_metrics: bool = False,
    ref_to_use: str = "refA",
    return_std_dev: bool = False,
) -> None:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    metric2color = {
        f"MaTESe-{ref_to_use}": "firebrick",
        f"NOISIFIED-MATESE-{ref_to_use}": "firebrick",
        "GEMBA-MQM-src": "darkblue",
        "NOISIFIED-GEMBA-MQM-src": "darkblue",
        f"XCOMET-Ensemble-{ref_to_use}": "darkgreen",
        "XCOMET-QE-Ensemble-src": "lightgreen",
        f"MetricX-23-{ref_to_use}": "darkorange",
        "MetricX-23-QE-src": "gold",
        "mbr-metricx-qe-src": "gray",
        "CometKiwi-src": "cornflowerblue",
        f"COMET-{ref_to_use}": "rebeccapurple",
        f"BLEURT-20-{ref_to_use}": "lightpink",
    }

    sentinels = [f"NOISIFIED-MATESE-{ref_to_use}", "NOISIFIED-GEMBA-MQM-src"]
    discrete = [f"MaTESe-{ref_to_use}", "GEMBA-MQM-src"]

    metric_names = [
        f"XCOMET-Ensemble-{ref_to_use}",
        "XCOMET-QE-Ensemble-src",
        f"MetricX-23-{ref_to_use}",
        "MetricX-23-QE-src",
        "mbr-metricx-qe-src",
        "CometKiwi-src",
        f"COMET-{ref_to_use}",
        f"BLEURT-20-{ref_to_use}",
        "GEMBA-MQM-src",
        f"MaTESe-{ref_to_use}",
        "NOISIFIED-GEMBA-MQM-src",
        f"NOISIFIED-MATESE-{ref_to_use}",
    ]

    def adjust_ranks(
        corr: Dict[str, Tuple[float, float]], metric_names: List[str]
    ) -> Dict[str, Tuple[float, int]]:
        sorted_items = sorted(
            [item for item in corr.items() if item[0] in metric_names],
            key=lambda item: item[1][0],
            reverse=True,
        )

        for idx, (key, value) in enumerate(sorted_items):
            sorted_items[idx] = (key, (value[0], idx + 1))

        return dict(sorted_items)

    best_tie_thresholds = [
        result.pop("metric_name2best_tie_threshold") for result in results
    ]
    metric_name2best_tie_thresholds = dict()
    for metric_name2best_tie_threshold in best_tie_thresholds:
        for metric_name, best_tie_threshold in metric_name2best_tie_threshold.items():
            if metric_name not in metric_name2best_tie_thresholds:
                metric_name2best_tie_thresholds[metric_name] = []
            metric_name2best_tie_thresholds[metric_name].append(
                best_tie_threshold[0] if return_std_dev else best_tie_threshold
            )  # 1st elem: avg, 2nd elem: std dev
    orig_thresholds = [[] for _ in metric_names]
    # min-max scaling
    thresholds = [[] for _ in metric_names]
    for metric_idx, metric_name in enumerate(metric_names):
        best_tie_thresholds = metric_name2best_tie_thresholds[metric_name]
        min_threshold, max_threshold = min(best_tie_thresholds), max(
            best_tie_thresholds
        )
        thresholds[metric_idx] = [
            (threshold - min_threshold) / (max_threshold - min_threshold)
            # the last threshold is used only for scaling, but not included in the picture
            for threshold in best_tie_thresholds[:-1]
        ]
        orig_thresholds[metric_idx] = best_tie_thresholds[:-1]

    # the last threshold is used only for scaling, but not included in the picture
    metrics_ties_dicts = [result.pop("metric_name2n_ties") for result in results[:-1]]
    metrics_ties = [[] for _ in metric_names]
    for metric_name2n_ties in metrics_ties_dicts:
        for metric_idx, metric_name in enumerate(metric_names):
            metrics_ties[metric_idx].append(
                metric_name2n_ties[metric_name][0]
                if return_std_dev
                else metric_name2n_ties[metric_name]
            )  # 1st elem: avg, 2nd elem: std dev

    # not taking the result of the last iteration as it is used only for scaling the thresholds
    orig_human_ties_proportions = [
        result.pop("human_ties_proportion") for result in results[:-1]
    ]
    human_ties_proportions = [
        proportion[0] if return_std_dev else proportion
        for proportion in orig_human_ties_proportions
    ]
    n_pairs_list = [result.pop("n_pairs") for result in results[:-1]]
    n_dev_rows_list, n_test_rows_list = [
        result.pop("n_dev_rows") for result in results[:-1]
    ], [result.pop("n_test_rows") for result in results[:-1]]
    test_set_human_ties_proportion = [
        result.pop("test_set_human_ties_proportion") for result in results[:-1]
    ]
    corrs = [adjust_ranks(corr, metric_names) for corr in results[:-1]]
    ranks = [[] for _ in metric_names]
    correlations = [[] for _ in metric_names]
    for corr_dict in corrs:
        for metric_idx, metric_name in enumerate(metric_names):
            rank = corr_dict[metric_name][1]
            correlation = corr_dict[metric_name][0]
            ranks[metric_idx].append(rank)
            correlations[metric_idx].append(correlation)

    cmap = plt.get_cmap("tab20")  # This colormap provides 20 distinct colors
    colors = [cmap(i) for i in range(len(metric_names))]
    # Overall percentage of ties on the x-axis
    x_values = human_ties_proportions

    def plot_data(
        lines: List,
        y_label: str,
        title: str,
        fig_name: str,
        data_type: str = "accuracies",
        fig_size: Tuple[int, int] = (6, 6),
        title_fontsize: int = 18,
        axis_labels_fontsize: int = 16,
        legend_fontsize: int = 13,
    ) -> None:
        plt.figure(figsize=fig_size)

        for i, line in enumerate(lines):
            if grey_continuous_metrics:
                color = (
                    "lightgray"
                    if metric_names[i] not in (discrete + sentinels)
                    else metric2color.get(metric_names[i], None)
                )
            else:
                color = metric2color.get(metric_names[i], "lightgray")
            zorder = 2 if metric_names[i] in metric2color else 1
            linestyle = "-" if metric_names[i] not in sentinels else "--"
            linewidth = 1 if metric_names[i] not in (discrete + sentinels) else 1.5
            plt.plot(
                x_values,
                line,
                label=metric_names[i],
                color=color,
                zorder=zorder,
                linestyle=linestyle,
                linewidth=linewidth,
            )

        plt.xticks(
            x_values, [f"{round(tie_proportion, 2)}" for tie_proportion in x_values]
        )
        plt.xlabel("Percentage of gold ties", fontsize=axis_labels_fontsize)
        plt.ylabel(y_label, fontsize=axis_labels_fontsize)
        plt.title(title, fontsize=title_fontsize)

        if data_type == "accuracies":
            pass

            # plt.legend(
            #     bbox_to_anchor=(1, 0), loc="lower right", fontsize=legend_fontsize
            # )
        else:
            plt.legend(
                bbox_to_anchor=(0.05, 0.95),
                loc="upper left",
                fontsize=str(legend_fontsize),
            )
        plt.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        plt.margins(x=0)
        plt.savefig(plots_dir_path / f"{fig_name}.pdf", format="pdf")

    plot_data(
        correlations,
        r"acc$_{\text{eq}}$",
        r"acc$_{\text{eq}}$ values for varying percentages of gold ties",
        f"accuracy-varying-ties-{lp}",
        data_type="accuracies",
    )
    plot_data(
        thresholds,
        r"Threshold $\epsilon$",
        r"Optimal $\epsilon$ for varying percentages of gold ties",
        f"thresholds-varying-ties-{lp}",
        "thresholds",
    )
    """
    plot_data(
        metrics_ties,
        r"Metric ties proportion",
        r"Proportion of metric ties for varying percentages of gold ties",
        "metric_ties-varying-ties",
        "metric ties",
    )
    plot_data(
        orig_thresholds,
        r"$\epsilon$ (without min-max scaling)",
        r"Optimal $\epsilon$ for varying percentages of gold ties",
        "orig-thresholds-varying-ties",
        "thresholds",
    )
    """

    print("\n")
    print("# pairs list:")
    pp(n_pairs_list)
    print("\n")
    print("# dev rows list:")
    pp(n_dev_rows_list)
    print("\n")
    print("# test rows list:")
    pp(n_test_rows_list)
    print("\n")
    print("Human ties proportions in dev set:")
    pp(orig_human_ties_proportions)
    print("\n")
    print("Human ties proportion in test set (not None only in the held-out exps):")
    pp(test_set_human_ties_proportion)
    print("\n")


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()

    metric_name2scores, metric_pairs_to_compare = dict(), []
    if args.metrics_to_evaluate_info_filepath is not None:
        metric_name2scores = get_metric_name2scores(
            args.metrics_to_evaluate_info_filepath, args.ref_to_use
        )

    testset = get_wmt_testset(args.testset_name, args.lp, True)

    accuracies = compute_accuracies_varying_ties(
        testset,
        args.ref_to_use,
        args.include_human,
        args.include_outliers,
        args.gold_name,
        args.primary_metrics,
        args.sample_rate,
        args.macro,
        args.seed,
        args.n_resampling,
        args.use_sampled_pairs_only_to_compute_eps,
        args.held_out_proportion,
        args.return_std_dev,
        args.return_n_rows,
        args.ranks_dir_path,
        metric_name2scores,
    )
    plot_accuracies(
        accuracies,
        args.plots_dir_path,
        args.lp,
        grey_continuous_metrics=args.grey_continuous_metrics,
        ref_to_use=args.ref_to_use,
        return_std_dev=args.return_std_dev,
    )
