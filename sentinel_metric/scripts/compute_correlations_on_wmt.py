import pickle
from argparse import ArgumentParser
from pathlib import Path
from pprint import pp
from typing import Optional, List, Dict, Tuple, Callable

import scipy.stats
from mt_metrics_eval import data, stats

from sentinel_metric.scripts.score import get_wmt_testset


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to compute correlations based on WMT data."
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
        "--k",
        type=int,
        default=1000,
        help="The number of resampling runs for statistical significance. Default: 1000.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=100,
        help="The size of blocks for 'early stopping' checks during resampling. Default: 100.",
    )
    parser.add_argument(
        "--pvalue",
        type=float,
        default=0.05,
        help="The p-value for the statistical significance test. Default: 0.05.",
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
        "--run-statistical-significance-for-item-grouping",
        action="store_true",
        help="Whether to run statistical significance test for correlations with 'item' grouping (it is very slow).",
    )
    parser.add_argument(
        "--run-statistical-significance-for-sys-grouping",
        action="store_true",
        help="Whether to run statistical significance test for correlations with 'sys' grouping (it is very slow).",
    )
    parser.add_argument(
        "--compute-only-seg-level-pearson-correlations",
        action="store_true",
        help="Whether to compute only seg-level Pearson correlations wrt human judgements.",
    )
    parser.add_argument(
        "--compute-only-seg-level-kendall-correlations",
        action="store_true",
        help="Whether to compute only seg-level KendallTau correlations wrt human judgements.",
    )
    parser.add_argument(
        "--print-also-in-tsv-and-latex-formats",
        action="store_true",
        help="Whether to print the final rankings also in tsv and LaTeX formats.",
    )
    return parser


def get_metric_name2scores(
    metrics_to_evaluate_filepath: Path, ref_to_use: str
) -> Dict[
    str, Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]
]:
    """Read the input file containing the required info for each metric and return a dictionary.

    Args:
        metrics_to_evaluate_filepath [Path]: Path to the file containing metrics info.
        ref_to_use [str]: Which reference to use for reference-based metrics.

    Returns:
        Dict[str, Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]]: Dictionary from metric
                                                                                              name to its scores.
    """
    metric_name2scores = dict()
    with open(metrics_to_evaluate_filepath) as metrics_file:
        for metric_info_line in metrics_file:
            info = metric_info_line.strip().split("\t")
            if len(info) != 3:
                raise ValueError(
                    f"Error during parsing the file {metrics_to_evaluate_filepath}, the line {info} should"
                    " contain 3 tab-separated elements: 'metric_name', 'is_refless' and 'output_scores_dir',"
                    " or 4 tab-separated elements: 'metric_name_1', 'is_metric_1_refless', 'metric_name_2'"
                    " and 'is_metric_2_refless'."
                )
            metric_name, is_refless, output_scores_dir = info
            output_scores_dir = (
                Path(output_scores_dir) if output_scores_dir != "None" else "None"
            )
            is_refless = is_refless.lower() == "yes"

            metric_name = (
                f"{metric_name}-src" if is_refless else f"{metric_name}-{ref_to_use}"
            )

            seg_scores, sys_scores = None, None
            if output_scores_dir != "None":
                with open(output_scores_dir / "seg_scores.pickle", "rb") as handle:
                    seg_scores = pickle.load(handle)
                with open(output_scores_dir / "sys_scores.pickle", "rb") as handle:
                    sys_scores = pickle.load(handle)

            if metric_name in metric_name2scores:
                raise ValueError(
                    f"The metric with name {metric_name} is present two times in the input file!"
                )
            metric_name2scores[metric_name] = (seg_scores, sys_scores)

    return metric_name2scores


def print_wmt_human_ratings_correlation_reports(
    testset: data.EvalSet,
    ref_to_use: str,
    include_human: bool,
    include_outliers: bool,
    gold_name: str,
    primary_metrics: bool,
    k: int,
    block_size: int,
    pvalue: float,
    sample_rate: float,
    macro: bool,
    run_statistical_significance_for_item_grouping: bool,
    run_statistical_significance_for_sys_grouping: bool,
    compute_only_seg_level_pearson_correlations: bool,
    compute_only_seg_level_kendall_correlations: bool,
    print_also_in_tsv_and_latex_formats: bool,
    metric_name2scores: Dict[
        str, Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]
    ],
) -> None:
    """Correlate the scores obtained by the metric models with the human ratings provided by WMT.

    Args:
        testset (data.EvalSet): The WMT test set to use.
        ref_to_use (str): Which human reference to use.
        include_human (bool): Whether to include 'human' systems (i.e., reference translations) among systems.
        include_outliers (bool): Whether to include systems considered to be outliers.
        gold_name (str): Which human ratings to use as gold scores.
        primary_metrics (bool): Whether to compare only metrics that have been designated as primary submissions.
        k (int): The number of resampling runs.
        block_size (int): The size of blocks for 'early stopping' checks during resampling.
        pvalue (float): The p-value for the significance test.
        sample_rate (float): Sample rate to pass to tau_optimization for 'KendallWithTiesOpt' (used only in wmt23).
        macro (bool): Whether to compute correlations with a plain average over row- or item-wise correlations.
        run_statistical_significance_for_item_grouping (bool): Whether to run statistical significance test for
                                                               correlations with 'item' grouping (it is very slow).
        run_statistical_significance_for_sys_grouping (bool) Whether to run statistical significance test for
                                                             correlations with 'sys' grouping (it is very slow).
        compute_only_seg_level_pearson_correlations (bool): Whether to compute only seg-level Pearson correlations wrt
                                                            human judgements.
        compute_only_seg_level_kendall_correlations (bool): Whether to compute only seg-level KendallTau correlations
                                                            wrt human judgements.
        print_also_in_tsv_and_latex_formats (bool): Whether to print the final rankings also in tsv and LaTeX formats.
        metric_name2scores (Dict[str, Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]]):
                            Dictionary from metric name to its scores.
    """
    psd = stats.PermutationSigDiffParams(block_size=block_size)

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

    def print_metric_comparison_from_corrs(
        metric_corrs: Dict[str, stats.Correlation],
        corr_fcn: Callable,
        average_by: str,
        level: str,
    ) -> None:
        level = level.capitalize()
        k_for_stat_sign = k
        if (
            average_by == "item" and not run_statistical_significance_for_item_grouping
        ) or (
            average_by == "sys" and not run_statistical_significance_for_sys_grouping
        ):
            k_for_stat_sign = 0

        if corr_fcn == stats.KendallWithTiesOpt:
            (
                corrs_and_ranks,
                sig_matrix,
                draws_index,
                draws_list,
                metric_name2best_tie_threshold,
            ) = data.CompareMetrics(
                metric_corrs,
                stats.KendallWithTiesOpt,
                average_by,
                macro,
                k_for_stat_sign,
                psd,
                pvalue,
                perm_test="pairs",
                return_metric_name2best_tie_threshold=True,
                sample_rate=sample_rate,
            )
            print("\n")
            print(
                f"{level}-level KendallWithTiesOpt corrs with '{average_by}' grouping strategy:"
            )
            data.PrintMetricComparison(
                corrs_and_ranks,
                sig_matrix,
                pvalue,
                testset,
                print_also_in_tsv_and_latex_formats=print_also_in_tsv_and_latex_formats,
            )
            print("\n")
            print("Best tie thresholds:")
            pp(metric_name2best_tie_threshold)
            print("\n")
        else:
            corr_fcn_name = (
                "Kendall Tau" if corr_fcn == scipy.stats.kendalltau else "Pearson"
            )
            corrs_and_ranks, sig_matrix, draws_index, draws_list = data.CompareMetrics(
                metric_corrs, corr_fcn, average_by, macro, k_for_stat_sign, psd, pvalue
            )
            print("\n")
            print(
                f"{level}-level {corr_fcn_name} corrs with {average_by} grouping strategy:"
            )
            data.PrintMetricComparison(
                corrs_and_ranks,
                sig_matrix,
                pvalue,
                testset,
                print_also_in_tsv_and_latex_formats=print_also_in_tsv_and_latex_formats,
            )
            print("\n")

    for corr_function in [
        scipy.stats.kendalltau,
        stats.KendallWithTiesOpt,
        scipy.stats.pearsonr,
    ]:
        if (
            compute_only_seg_level_pearson_correlations
            and corr_function != scipy.stats.pearsonr
        ) or (
            compute_only_seg_level_kendall_correlations
            and corr_function != scipy.stats.kendalltau
        ):
            continue
        if corr_function != stats.KendallWithTiesOpt:
            print_metric_comparison_from_corrs(corrs, corr_function, "none", "seg")
        print_metric_comparison_from_corrs(corrs, corr_function, "item", "seg")
        if corr_function != stats.KendallWithTiesOpt:
            print_metric_comparison_from_corrs(corrs, corr_function, "sys", "seg")

    if (
        not compute_only_seg_level_pearson_correlations
        and not compute_only_seg_level_kendall_correlations
    ):
        corrs = data.GetCorrelations(
            testset,
            "sys",
            {ref_to_use},
            set(),
            include_human,
            include_outliers,
            gold_name,
            primary_metrics,
            None,
            extern_metrics=sys_extras,
        )
        print_metric_comparison_from_corrs(corrs, scipy.stats.pearsonr, "none", "sys")


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()

    metric_name2scores = dict()
    if args.metrics_to_evaluate_info_filepath is not None:
        metric_name2scores = get_metric_name2scores(
            args.metrics_to_evaluate_info_filepath, args.ref_to_use
        )

    testset = get_wmt_testset(args.testset_name, args.lp, True)

    print_wmt_human_ratings_correlation_reports(
        testset,
        args.ref_to_use,
        args.include_human,
        args.include_outliers,
        args.gold_name,
        args.primary_metrics,
        args.k,
        args.block_size,
        args.pvalue,
        args.sample_rate,
        args.macro,
        args.run_statistical_significance_for_item_grouping,
        args.run_statistical_significance_for_sys_grouping,
        args.compute_only_seg_level_pearson_correlations,
        args.compute_only_seg_level_kendall_correlations,
        args.print_also_in_tsv_and_latex_formats,
        metric_name2scores,
    )
