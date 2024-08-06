import pickle
from argparse import ArgumentParser
from pathlib import Path

from mt_metrics_eval import data, stats, tasks, meta_info
from typing import Dict, List, Tuple

wmt23_lps = ["en-de", "he-en", "zh-en"]


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command to compute the WMT-23 final ranking.")
    parser.add_argument(
        "--metrics-to-evaluate-info-filepath",
        type=Path,
        help="Path to the file containing the info for metrics to evaluate.",
    )
    parser.add_argument(
        "--metrics-outputs-path",
        type=Path,
        help="Path to the directory containing the scores returned by the metrics for all language pairs.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1000,
        help="The number of resampling runs for statistical significance. Default: 1000.",
    )
    parser.add_argument(
        "--only-seg-level",
        action="store_true",
        help="Whether to compute the ranking considering only the segment-level tasks.",
    )
    parser.add_argument(
        "--item-for-seg-level-pearson",
        action="store_true",
        help="Whether to compute segment-level Pearson correlation with 'item' grouping strategy instead of 'none'.",
    )
    parser.add_argument(
        "--only-pearson",
        action="store_true",
        help="Whether to compute the ranking considering only the Pearson correlation.",
    )
    return parser


def get_metric_name2lp_scores(
    metrics_to_evaluate_info_filepath: Path, metrics_outputs_filepath: Path
) -> Tuple[Dict[str, Dict[str, Dict[str, List[float]]]], Dict[str, Dict[str, str]]]:
    """Read the input files and return dictionary with the scores for each metric and a dictionary with the used refs.

    Args:
        metrics_to_evaluate_info_filepath (Path): Path to the file containing the info for metrics to evaluate.
        metrics_outputs_filepath (Path): Path to the directory containing the scores returned by the metrics for all
                                         language pairs.

    Returns:
        Tuple[Dict[str, Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, str]]]: Dicts with scores and refs.
    """
    metric_name2lp_scores, metric_name2ref_to_use = dict(), dict()

    with open(metrics_to_evaluate_info_filepath) as metrics_info_file:
        for line in metrics_info_file:
            info = line.strip().split("\t")
            if len(info) != 5:
                raise ValueError(
                    f"Expected 5 tab-separated values, got {len(info)}: {info}"
                )

            metric_name, scores_dir_name, zh_en_ref, en_de_ref, he_en_ref = info
            metric_name2ref_to_use[metric_name] = {
                "zh-en": zh_en_ref,
                "en-de": en_de_ref,
                "he-en": he_en_ref,
            }

            metric_name2lp_scores[metric_name] = dict()
            for lp in wmt23_lps:
                metric_name2lp_scores[metric_name][lp] = dict()
                for level, scores_filename in [
                    ("seg", "seg_scores.pickle"),
                    ("sys", "sys_scores.pickle"),
                ]:
                    with open(
                        metrics_outputs_filepath
                        / lp
                        / scores_dir_name
                        / scores_filename,
                        "rb",
                    ) as handle:
                        metric_name2lp_scores[metric_name][lp][level] = pickle.load(
                            handle
                        )

    return metric_name2lp_scores, metric_name2ref_to_use


def compute_final_wmt_ranking_command() -> None:
    """Command to compute the final WMT-23 ranking."""
    parser = read_arguments()
    args = parser.parse_args()

    metric_name2lp_scores, metric_name2ref_to_use = get_metric_name2lp_scores(
        args.metrics_to_evaluate_info_filepath, args.metrics_outputs_path
    )
    compute_final_wmt_ranking(
        metric_name2lp_scores,
        metric_name2ref_to_use,
        args.k,
        args.only_seg_level,
        args.item_for_seg_level_pearson,
        args.only_pearson,
    )


def compute_final_wmt_ranking(
    metric_name2lp_scores: Dict[str, Dict[str, Dict[str, List[float]]]],
    metric_name2ref_to_use: Dict[str, Dict[str, str]],
    k: int,
    only_seg_level: bool,
    item_for_seg_level_pearson: bool,
    only_pearson: bool,
) -> None:
    """Compute the final WMT-23 ranking.

    Args:
        metric_name2lp_scores (Dict[str, Dict[str, Dict[str, List[float]]]]): Dictionary with the scores for each
                                                                              metric.
        metric_name2ref_to_use (Dict[str, Dict[str, str]]): Dictionary with the used refs.
        k (int): The number of resampling runs for statistical significance.
        only_seg_level (bool): Whether to compute the ranking considering only the segment-level tasks.
        item_for_seg_level_pearson (bool): Whether to compute segment-level Pearson correlation with 'item' grouping
                                           strategy instead of 'none'.
        only_pearson (bool): Whether to compute the ranking considering only the Pearson correlation.
    """
    evs_dict = {("wmt23", lp): data.EvalSet("wmt23", lp, True) for lp in wmt23_lps}

    for lp in wmt23_lps:
        evs = evs_dict[("wmt23", lp)]
        for metric_name, lp_scores in metric_name2lp_scores.items():
            seg_scores, sys_scores = lp_scores[lp]["seg"], lp_scores[lp]["sys"]
            refs = {metric_name2ref_to_use[metric_name][lp]}
            refs = refs if refs != {"src"} else set()
            evs.AddMetric(metric_name, refs, "seg", seg_scores, replace=True)
            if not only_seg_level:
                evs.AddMetric(metric_name, refs, "sys", sys_scores, replace=True)

    for evs in evs_dict.values():
        evs.SetPrimaryMetrics(evs.primary_metrics | set(metric_name2lp_scores))

    wmt23_tasks, wts = tasks.WMT23(
        wmt23_lps,
        k=k,
        only_seg_level=only_seg_level,
        item_for_seg_level_pearson=item_for_seg_level_pearson,
        only_pearson=only_pearson,
    )
    new_results = wmt23_tasks.Run(eval_set_dict=evs_dict)
    avg_corrs = new_results.AverageCorrs(wts)
    table = new_results.Table(
        metrics=list(avg_corrs),
        initial_column=avg_corrs,
        initial_column_header="avg-corr",
        attr_list=["lang", "level", "corr_fcn"],
        nicknames={"KendallWithTiesOpt": "acc-t"},
        fmt="text",
        baselines_metainfo=meta_info.WMT23,
    )
    print(table)

    print("\n\n\n")

    table = new_results.Table(
        metrics=list(avg_corrs),
        initial_column=avg_corrs,
        initial_column_header="avg-corr",
        attr_list=["lang", "level", "corr_fcn"],
        nicknames={"KendallWithTiesOpt": "acc-t"},
        fmt="latex",
        baselines_metainfo=meta_info.WMT23,
    )
    print(table)


if __name__ == "__main__":
    compute_final_wmt_ranking_command()
