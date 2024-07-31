from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np

from sentinel_metric.scripts.score import get_wmt_testset


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to add Gaussian noise to the scores given by a WMT metric specified in input."
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        help="Name of the WMT metric to use.",
    )
    parser.add_argument(
        "--gaussian-noise-mean",
        type=float,
        default=0,
        help="Mean of the Gaussian noise to add to the metric scores. Defaults to 0.",
    )
    parser.add_argument(
        "--gaussian-noise-std-dev",
        default=0.1,
        type=float,
        help="Standard deviation to use for the Gaussian noise. Default: 0.1.",
    )
    parser.add_argument(
        "--seed-for-noise",
        type=int,
        default=42,
        help="Seed to use to generate the random Gaussian noise. Defaults to 42.",
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
        "--is-ref-less",
        action="store_true",
        help="Whether the metric is reference-less.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Path to the directory where to save scores.",
    )
    return parser


def add_noise_to_metric_scores(
    metric_name: str,
    gaussian_noise_mean: float,
    gaussian_noise_std_dev: float,
    testset_name: str,
    lp: str,
    ref_to_use: str,
    is_ref_less: bool,
    out_dir: Path,
) -> None:
    """Adds random Gaussian noise to the scores given by a WMT metric specified in input.

    Args:
        metric_name (str): Name of the WMT metric to use.
        gaussian_noise_mean (float): Mean of the Gaussian noise to add to the metric scores.
        gaussian_noise_std_dev (float): Standard deviation to use for the Gaussian noise.
        testset_name (str): Name of the WMT test set to use.
        lp (str): Language pair to consider in the test set passed in input.
        ref_to_use (str): Which human reference to use (it will be used iff the metric is ref-based). It must be like
                          refA, refB, etc.
        is_ref_less (bool): Whether the metric is reference-less.
        out_dir (Path): Path to the directory where to save scores.
    """
    testset = get_wmt_testset(testset_name, lp, True)
    metric_scores = testset.Scores(
        "seg", f"{metric_name}-src" if is_ref_less else f"{metric_name}-{ref_to_use}"
    )

    new_metric_sys2seg_scores, new_metric_sys2score = dict(), dict()
    for sys_name, seg_scores in metric_scores.items():
        noise = np.random.normal(
            gaussian_noise_mean, gaussian_noise_std_dev, len(seg_scores)
        )
        new_metric_sys2seg_scores[sys_name] = (np.array(seg_scores) + noise).tolist()
        new_metric_sys2score[sys_name] = [
            sum(new_metric_sys2seg_scores[sys_name])
            / len(new_metric_sys2seg_scores[sys_name])
        ]

    with open(out_dir / "seg_scores.pickle", "wb") as handle:
        pickle.dump(new_metric_sys2seg_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_dir / "sys_scores.pickle", "wb") as handle:
        pickle.dump(new_metric_sys2score, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = read_arguments()
    args = parser.parse_args()

    np.random.seed(args.seed_for_noise)
    add_noise_to_metric_scores(
        args.metric_name,
        args.gaussian_noise_mean,
        args.gaussian_noise_std_dev,
        args.testset_name,
        args.lp,
        args.ref_to_use,
        args.is_ref_less,
        args.out_dir,
    )
