import json
import warnings
from argparse import ArgumentParser
from pathlib import Path
import pickle

from typing import List, Dict, Union, Optional

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
        read_stored_metric_scores (bool): Read stored scores for automatic metrics for this dataset. Defaults to False.

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
    parser = ArgumentParser(description="Command to score with a given metric model.")
    parser.add_argument(
        "--sentinel-metric-model-name",
        type=str,
        help="String that identifies the sentinel metric model to use from Hugging Face.",
    )
    parser.add_argument(
        "--sentinel-metric-model-checkpoint-path",
        type=Path,
        help="Path to the sentinel metric model checkpoint to use. If passed, the '--sentinel-metric-model-name' "
        "input arg will be ignored.",
    )
    parser.add_argument(
        "--sentinel-metric-model-class-identifier",
        type=str,
        default="sentinel_regression_metric",
        help="String that identifies the sentinel metric model class with which load the weights.",
    )
    parser.add_argument(
        "--comet-metric-model-name",
        type=str,
        help="String that identifies the COMET metric model to use from Hugging Face. If passed, the above arguments "
        "will be ignored.",
    )
    parser.add_argument(
        "--comet-metric-model-checkpoint-path",
        type=Path,
        help="Path to the COMET metric model checkpoint to use. If passed, the above arguments will be ignored.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Strictly enforce the matching between the keys for the state dict during the metric model load.",
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
        help="Batch size to use when running inference with the given metric model. Default: 8.",
    )
    parser.add_argument(
        "--testset-name",
        type=str,
        help="Name of the WMT test set to use.",
    )
    parser.add_argument(
        "--lp",
        type=str,
        help="Language pair to consider in the WMT test set passed in input.",
    )
    parser.add_argument(
        "--ref-to-use",
        type=str,
        help="Which human reference to use for the input WMT test set (it will be used iff the metric is ref-based). It"
        " must be like refA, refB, etc.",
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        help="Whether to include 'human' systems (i.e., reference translations) among systems to be scored in the "
        "input WMT test set.",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Whether to include systems considered to be outliers in the input WMT test set.",
    )
    parser.add_argument(
        "--include-ref-to-use",
        action="store_true",
        help="Whether to include the 'ref_to_use' system among systems to be scored in the input WMT test set.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="If passed, it limits the scoring to only the specified domain in the input WMT test set.",
    )
    parser.add_argument(
        "--csv-data-path",
        type=Path,
        help="Path to the .csv file containing the data to score.",
    )
    parser.add_argument(
        "--computed-scores-column-name",
        type=str,
        help="Name of the column where to save the computed scores in the input .csv data.",
    )
    parser.add_argument(
        "-s",
        "--sources",
        type=Path,
        help="Path to the file containing the source sentences.",
    )
    parser.add_argument(
        "-t",
        "--translations",
        type=Path,
        nargs="*",
        help="Path to the file containing the candidate translations.",
    )
    parser.add_argument(
        "-r",
        "--references",
        type=Path,
        help="Path to the file containing the reference translations.",
    )
    parser.add_argument(
        "--only-system",
        action="store_true",
        help="Whether to print only the final system score.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        help="Path to the directory where to save the pickle dictionaries containing scores, or directly to the new "
        ".csv file containing the added scores column.",
    )
    parser.add_argument(
        "--to-json",
        type=Path,
        help="Path to the json file where to save input data together with predicted scores.",
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        help="String name of the metric. It will be used only for the output json, if passed.",
    )
    return parser


def score_command() -> None:
    """Command to score with a given metric model."""
    parser = read_arguments()
    args = parser.parse_args()

    if (args.testset_name is not None) + (args.csv_data_path is not None) + (
        args.sources is not None
        or args.translations is not None
        or args.references is not None
    ) != 1:
        parser.error(
            "Exactly one of '--testset-name', '--csv-data-path', or '--sources'/'--translations'/'--references' must "
            "be passed!"
        )

    if args.testset_name is not None and (args.lp is None or args.ref_to_use is None):
        parser.error(
            "If '--testset-name' is passed, '--lp' and '--ref-to-use' must also be passed!"
        )

    if args.csv_data_path is not None and (
        args.computed_scores_column_name is None or args.out_path is None
    ):
        parser.error(
            "If '--csv-data-path' is passed, '--computed-scores-column-name' and '--out-path' must also be passed!"
        )

    if args.comet_metric_model_checkpoint_path is not None:
        metric_model = comet.load_from_checkpoint(
            args.comet_metric_model_checkpoint_path, strict=args.strict_load
        )
    elif args.comet_metric_model_name is not None:
        metric_model_path = comet.download_model(args.comet_metric_model_name)
        metric_model = comet.load_from_checkpoint(
            metric_model_path, strict=args.strict_load
        )
    elif args.sentinel_metric_model_checkpoint_path:
        metric_model = sentinel_metric.load_from_checkpoint(
            args.sentinel_metric_model_checkpoint_path,
            strict=args.strict_load,
            class_identifier=args.sentinel_metric_model_class_identifier,
        )
    else:
        if args.sentinel_metric_model_name is None:
            parser.error("No metric model specified in input!")

        metric_model_path = sentinel_metric.download_model(
            args.sentinel_metric_model_name
        )
        metric_model = sentinel_metric.load_from_checkpoint(
            metric_model_path,
            strict=args.strict_load,
            class_identifier=args.sentinel_metric_model_class_identifier,
        )

    score_with_metric_model(
        metric_model,
        args.gpus,
        args.batch_size,
        args.testset_name,
        args.lp,
        args.include_human,
        args.include_ref_to_use,
        args.include_outliers,
        args.only_system,
        args.ref_to_use,
        args.domain,
        args.csv_data_path,
        args.computed_scores_column_name,
        args.sources,
        args.translations,
        args.references,
        args.out_path,
        args.to_json,
        args.metric_name,
    )


def score_with_metric_model(
    metric_model: Union[RegressionMetricModel, CometModel],
    gpus: int,
    batch_size: int,
    testset_name: str,
    lp: str,
    include_human: bool,
    include_ref_to_use: bool,
    include_outliers: bool,
    only_system: bool,
    ref_to_use: Optional[str] = None,
    domain: Optional[str] = None,
    csv_data_path: Optional[Path] = None,
    computed_scores_column_name: Optional[str] = None,
    sources: Optional[Path] = None,
    translations: Optional[List[Path]] = None,
    references: Optional[Path] = None,
    out_path: Optional[Path] = None,
    to_json: Optional[Path] = None,
    metric_name: Optional[str] = None,
) -> None:
    """Scores with a given metric model.

    Args:
        metric_model (Union[RegressionMetricModel, CometModel]): Metric model to use for scoring.
        gpus (int): Number of GPUs to use for inference.
        batch_size (int): Batch size to use when running inference with the metric model.
        testset_name (str): Name of the WMT test set to use.
        lp (str): Language pair to consider in the test set passed in input.
        include_human (bool): Whether to include 'human' systems among systems to be scored.
        include_ref_to_use (bool): Whether to include the 'ref_to_use' system among systems to be scored.
        include_outliers (bool): Whether to include systems considered to be outliers.
        only_system (bool): Whether to print only the final system score.
        ref_to_use (Optional[str]): Which human reference to use. It must be like refA, refB, etc. Defaults to None.
        domain (Optional[str]): If passed, it limits the scoring to only the specified domain. Defaults to None.
        csv_data_path (Optional[Path]): Path to the .csv file containing the data to score. Defaults to None.
        computed_scores_column_name (Optional[str]): Name of the column for the computed scores. Defaults to None.
        sources (Optional[Path]): Path to the file containing the source sentences. Defaults to None.
        translations (Optional[List[Path]]): Path to the file containing the candidate translations. Defaults to None.
        references (Optional[Path]): Path to the file containing the reference translations. Defaults to None.
        out_path (Optional[Path]): Directory where to save scores, or directly to the new .csv file. Defaults to None.
        to_json (Optional[Path]): Path to the json file where to save input data with scores. Defaults to None.
        metric_name (Optional[str]): String name of the metric to report in the output json. Defaults to None.
    """

    def create_input_data_for_metric_model(
        src_sents: List[str], cand_sents: List[str], ref_sents: List[str]
    ) -> List[Dict[str, str]]:
        """Create the input data for the metric model.

        Args:
            src_sents (List[str]): Source sentences.
            cand_sents (List[str]): Candidate translation sentences.
            ref_sents (List[str]): Reference translation sentences.

        Returns:
            List[Dict[str, str]]: Input data for the metric model.
        """
        input_data = []
        # Determine the maximum length to handle different lengths safely (sentinel input may contain a single sent)
        max_len = max(len(src_sents), len(cand_sents), len(ref_sents))

        for i in range(max_len):
            data_dict = dict()
            if i < len(src_sents):
                data_dict["src"] = src_sents[i]
            if i < len(cand_sents):
                data_dict["mt"] = cand_sents[i]
            if i < len(ref_sents):
                data_dict["ref"] = ref_sents[i]

            if data_dict:
                input_data.append(data_dict)

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
        print(f"# samples to score in the input .csv data = {len(cand_sents)}.")
        print("\n")

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
        src_sents, ref_sents = [], []
        sys2outputs = dict()
        n_candidates_to_score = 0

        if sources is not None or translations is not None or references is not None:
            if sources is not None:
                with open(sources, encoding="utf-8") as fp:
                    src_sents = [line.strip() for line in fp.readlines()]
                    sys2outputs["SOURCE"] = src_sents
            if translations is not None:
                for path in translations:
                    with open(path, encoding="utf-8") as fp:
                        if path.name in sys2outputs:
                            warnings.warn(
                                f"Filename {path.name} appears multiple times (filenames are used as system "
                                f"names). The last occurrence will overwrite the previous ones."
                            )
                        sys2outputs[path.name] = [
                            line.strip() for line in fp.readlines()
                        ]
                        n_candidates_to_score += len(sys2outputs[path.name])
            if references is not None:
                with open(references, encoding="utf-8") as fp:
                    ref_sents = [line.strip() for line in fp.readlines()]
                    sys2outputs["REFERENCE"] = ref_sents

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
                f"# MT systems to score in {testset_name} for {lp} lp = {len(sys2outputs)}."
            )
            if domain is not None:
                print(f"Domain: {domain}.")
            else:
                print("No domain is specified.")
            print("\n")

        print("\n")
        print(
            f"# input source sentences: {len(src_sents)}\t# input candidate translations: {n_candidates_to_score}\t"
            f"# input reference translations: {len(ref_sents)}."
        )
        print("\n")

        sys2seg_scores, sys2score, sys2scored_data = dict(), dict(), dict()
        max_scores_len = 0
        for sys_name, cand_sents in sys2outputs.items():
            metric_model_input_data = create_input_data_for_metric_model(
                src_sents, cand_sents, ref_sents
            )
            sys2scored_data[sys_name] = metric_model_input_data
            metric_model_output = metric_model.predict(
                metric_model_input_data,
                batch_size=batch_size,
                gpus=gpus,
            )

            if len(metric_model_output["scores"]) > max_scores_len:
                max_scores_len = len(metric_model_output["scores"])

            sys2seg_scores[sys_name] = metric_model_output["scores"]
            sys2score[sys_name] = [metric_model_output["system_score"]]

        metric_name = (
            f"{metric_name}_score" if metric_name is not None else "metric_score"
        )
        print("\n")
        for seg_idx in range(max_scores_len):
            for sys_name, scored_data in sys2scored_data.items():
                if seg_idx < len(sys2seg_scores[sys_name]):
                    sys2scored_data[sys_name][seg_idx][metric_name] = sys2seg_scores[
                        sys_name
                    ][seg_idx]
                    if not only_system:
                        print(
                            f"MT system: {sys_name}\tSegment idx: {seg_idx}\tMetric segment score: "
                            f"{round(sys2seg_scores[sys_name][seg_idx], 4)}."
                        )

        print("\n")
        for sys_name, score in sys2score.items():
            print(f"MT system: {sys_name}\tMetric system score: {round(score[0], 4)}.")
        print("\n")

        if to_json is not None:
            with open(to_json, "w", encoding="utf-8") as outfile:
                json.dump(sys2scored_data, outfile, ensure_ascii=False, indent=4)

        if out_path is not None:
            with open(out_path / "seg_scores.pickle", "wb") as handle:
                pickle.dump(sys2seg_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(out_path / "sys_scores.pickle", "wb") as handle:
                pickle.dump(sys2score, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    score_command()
