from argparse import ArgumentParser
from pathlib import Path
import pickle
from pprint import pp

from typing import List, Dict, Union, Sequence, Optional

import numpy as np
import pandas as pd

import sentinel_metric
from sentinel_metric.models import RegressionMetricModel

import comet
from comet.models import CometModel

from mt_metrics_eval import data


def get_wmt_testset(
    testset_name: str, lp: str, read_stored_metric_scores: bool = False
) -> data.EvalSet:
    """Return the WMT test set defined by the input parameters.

    Args:
        testset_name (str): Name of the WMT test set to use.
        lp (str): Language pair to consider in the test set passed in input.
        read_stored_metric_scores (bool): Read stored scores for automatic metrics for this dataset.

    Returns:
        data.EvalSet: WMT test set.
    """
    testset = data.EvalSet(testset_name, lp, read_stored_metric_scores)

    nsegs = len(testset.src)
    nsys = len(testset.sys_names)
    nmetrics = len(testset.metric_basenames)
    gold_seg = testset.StdHumanScoreName("seg")
    nrefs = len(testset.ref_names)
    std_ref = testset.std_ref

    print("\n")
    print(f"lp = {lp}.")
    print(f"# segs = {nsegs}.")
    print(f"# systems = {nsys}.")
    print(f"# metrics = {nmetrics}.")
    print(f"Std annotation type = {gold_seg}.")
    print(f"# refs = {nrefs}.")
    print(f"std ref = {std_ref}.")
    print("\n")

    return testset


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to score the WMT candidate translations with a given metric model."
    )
    parser.add_argument(
        "--metric-model-checkpoint-path",
        type=Path,
        help="Path to the metric model checkpoint to test on WMT.",
    )
    parser.add_argument(
        "--metric-model-class-identifier",
        type=str,
        default="sentinel_regression_metric",
        help="String that identifies the metric model class with which load the pre-trained weights.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Strictly enforce that the keys in metric_model_checkpoint_path match the keys returned by the state "
        "dict.",
    )
    parser.add_argument(
        "--comet-metric-model-name",
        type=str,
        help="String that identifies the COMET metric model to eventually download and use from HuggingFace. If passed,"
        " the above arguments except --strict-load will be ignored.",
    )
    parser.add_argument(
        "--comet-metric-model-checkpoint-path",
        type=Path,
        help="Path to the COMET metric model checkpoint to test on WMT. If passed, the above arguments except "
        "--strict-load will be ignored.",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs to use for inference. Default: 1.",
    )
    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="Batch size to use when running inference with the metric model. Default: 8.",
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
        help="Whether to include 'human' systems (i.e., reference translations) among systems to be scored.",
    )
    parser.add_argument(
        "--include-ref-to-use",
        action="store_true",
        help="Whether to include the 'ref_to_use' system among systems to be scored.",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Whether to include systems considered to be outliers.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="If passed, it limits the scoring to only the candidate translations that are in the specified domain.",
    )
    parser.add_argument(
        "--csv-data-path",
        type=Path,
        help="Path to the .csv file containing the data to score. If passed, the above arguments about which data to "
        "score will be ignored, and the input .csv data will be scored.",
    )
    parser.add_argument(
        "--computed-scores-column-name",
        type=str,
        help="Name of the column where to save the computed scores in the input .csv data.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        required=True,
        help="Path to the directory where to save scores, or directly to the new .csv file with the added scores "
        "column.",
    )
    return parser


def score_candidates(
    metric_model: Union[RegressionMetricModel, CometModel],
    gpus: int,
    batch_size: int,
    testset_name: str,
    lp: str,
    ref_to_use: str,
    include_human: bool,
    include_ref_to_use: bool,
    include_outliers: bool,
    out_path: Path,
    domain: Optional[str] = None,
    csv_data_path: Optional[Path] = None,
    computed_scores_column_name: Optional[str] = None,
) -> None:
    """Scores the WMT candidate translations with a given metric model.

    Args:
        metric_model (Union[RegressionMetricModel, CometModel]): Metric model to use for scoring.
        gpus (int): Number of GPUs to use for inference.
        batch_size (int): Batch size to use when running inference with the metric model.
        testset_name (str): Name of the WMT test set to use.
        lp (str): Language pair to consider in the test set passed in input.
        ref_to_use (str): Which human reference to use. It must be like refA, refB, etc.
        include_human (bool): Whether to include 'human' systems among systems to be scored.
        include_ref_to_use (bool): Whether to include the 'ref_to_use' system among systems to be scored.
        include_outliers (bool): Whether to include systems considered to be outliers.
        out_path (Path): Path to the directory where to save scores, or directly to the new .csv file.
        domain (Optional[str]): If passed, it limits the scoring to only the specified domain. Defaults to None.
        csv_data_path (Optional[Path]): Path to the .csv file containing the data to score. Defaults to None.
        computed_scores_column_name (Optional[str]): Name of the column for the computed scores. Defaults to None.
    """

    def create_input_data_for_metric_model(
        src_sents: List[str], cand_sents: List[str], ref_sents: List[str]
    ) -> List[Dict[str, str]]:
        input_data = []

        assert len(src_sents) == len(cand_sents) == len(ref_sents)
        for input_sents in [src_sents, ref_sents]:
            assert all(sent is not None and len(sent) > 0 for sent in input_sents)

        for src, cand, ref in zip(src_sents, cand_sents, ref_sents):
            input_data.append({"src": src, "mt": cand, "ref": ref})

        return input_data

    if csv_data_path is not None:
        dataset_to_score = pd.read_csv(csv_data_path)
        src_sents, cand_sents, ref_sents = (
            dataset_to_score["src"].tolist(),
            dataset_to_score["mt"].tolist(),
            dataset_to_score["ref"].tolist(),
        )
        assert len(src_sents) == len(cand_sents) == len(ref_sents)

        print("\n")
        print(f"# candidates to score in the input .csv data = {len(cand_sents)}.")
        print("\n")

        assert all(cand is not None for cand in cand_sents)
        metric_model_input_data = create_input_data_for_metric_model(
            src_sents, cand_sents, ref_sents
        )
        metric_model_output = metric_model.predict(
            metric_model_input_data,
            batch_size=batch_size,
            gpus=gpus,
        )

        seg_scores = metric_model_output["scores"]
        assert len(seg_scores) == len(cand_sents)
        assert (score is not None for score in seg_scores)
        dataset_to_score[computed_scores_column_name] = seg_scores
        dataset_to_score.to_csv(out_path, index=False)

    else:
        testset = get_wmt_testset(testset_name, lp)

        systems_to_discard = set()
        if not include_ref_to_use:
            systems_to_discard.add(ref_to_use)
        if not include_human:
            systems_to_discard = systems_to_discard.union(testset.human_sys_names)
        if not include_outliers:
            systems_to_discard = systems_to_discard.union(testset.outlier_sys_names)

        domains_per_seg = testset.DomainsPerSeg()
        src_sents, ref_sents = [], []
        sys2outputs = dict()
        n_candidates_to_score = 0
        for sys, candidates in testset.sys_outputs.items():
            if sys in systems_to_discard:
                continue
            assert (
                len(candidates)
                == len(testset.src)
                == len(domains_per_seg)
                == len(testset.all_refs[ref_to_use])
            )
            assert all(candidate is not None for candidate in candidates)
            if len(src_sents) == 0:
                sys2outputs[sys], src_sents, ref_sents = zip(
                    *[
                        (candidate, src_sent, ref_sent)
                        for candidate, src_sent, ref_sent, d in zip(
                            candidates,
                            testset.src,
                            testset.all_refs[ref_to_use],
                            domains_per_seg,
                        )
                        if domain is None or d == domain
                    ]
                )
            else:
                sys2outputs[sys] = [
                    candidate
                    for candidate, d in zip(candidates, domains_per_seg)
                    if domain is None or d == domain
                ]
            n_candidates_to_score += len(sys2outputs[sys])
        print("\n")
        print(
            f"# MT systems to score in {testset_name} for lp {lp} = {len(sys2outputs)}."
        )
        if domain is not None:
            print(f"The domain is: {domain}.")
        else:
            print("No domain is specified.")
        print(f"# candidates to score = {n_candidates_to_score}.")
        print("\n")

        sys2seg_scores, sys2score = dict(), dict()
        for sys_name, cand_sents in sys2outputs.items():
            assert len(src_sents) == len(ref_sents) == len(cand_sents)

            metric_model_input_data = create_input_data_for_metric_model(
                src_sents, cand_sents, ref_sents
            )
            metric_model_output = metric_model.predict(
                metric_model_input_data,
                batch_size=batch_size,
                gpus=gpus,
            )
            sys2seg_scores[sys_name] = metric_model_output["scores"]
            sys2score[sys_name] = [metric_model_output["system_score"]]

        with open(out_path / "seg_scores.pickle", "wb") as handle:
            pickle.dump(sys2seg_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out_path / "sys_scores.pickle", "wb") as handle:
            pickle.dump(sys2score, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()
    metric_model = None
    if args.comet_metric_model_checkpoint_path is not None:
        metric_model = comet.load_from_checkpoint(
            args.comet_metric_model_checkpoint_path, strict=args.strict_load
        )
    elif args.comet_metric_model_name is not None:
        metric_model_path = comet.download_model(args.comet_metric_model_name)
        metric_model = comet.load_from_checkpoint(
            metric_model_path, strict=args.strict_load
        )
    else:
        metric_model = sentinel_metric.load_from_checkpoint(
            args.metric_model_checkpoint_path,
            strict=args.strict_load,
            class_identifier=args.metric_model_class_identifier,
        )

    score_candidates(
        metric_model,
        args.gpus,
        args.batch_size,
        args.testset_name,
        args.lp,
        args.ref_to_use,
        args.include_human,
        args.include_ref_to_use,
        args.include_outliers,
        args.out_path,
        args.domain,
        args.csv_data_path,
        args.computed_scores_column_name,
    )
