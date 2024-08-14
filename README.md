<div align="center">

<h1 style="font-family: 'Arial', sans-serif; font-size: 28px; font-weight: bold; color: #f0f0f0;">
    🛡️ Guardians of the Machine Translation Meta-Evaluation:<br>
    Sentinel Metrics Fall In!
</h1>

[![Conference](https://img.shields.io/badge/ACL-2024-4b44ce
)](https://2024.aclweb.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://2024.aclweb.org/program/main_conference_papers/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FCD21D)](https://huggingface.co/collections/sapienzanlp/mt-sentinel-metrics-66ab643b32aab06f3157e5c1)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://github.com/Lightning-AI/lightning)
[![](https://shields.io/badge/-COMET-2F4DC1?style=flat&logo=github&labelColor=gray)](https://github.com/Unbabel/COMET)
[![](https://shields.io/badge/-MT%20Metrics%20Eval-green?style=flat&logo=github&labelColor=gray)](https://github.com/google-research/mt-metrics-eval)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

</div>

## Setup

The code in this repo requires Python 3.10 or higher. We recommend creating a new [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment as follows:

```bash
conda create -n guardians-mt-eval python=3.10
conda activate guardians-mt-eval val
pip install --upgrade pip
pip install -e .
```

## Data

We trained the sentinel metrics using the Direct Assessments (DA) and Multidimensional Quality Metrics (MQM) human annotations downloaded from the [COMET Github repository](https://github.com/Unbabel/COMET/tree/master/data).

## Models

We trained the following sentinel metrics:

| HF Model Name                                                                           | Input                 | Training Data               |
|-----------------------------------------------------------------------------------------|-----------------------|---------------              |
| [`sapienzanlp/sentinel-src-da`](https://huggingface.co/sapienzanlp/sentinel-src-da)     | Source text           | DA WMT17-20                 |
| [`sapienzanlp/sentinel-src-mqm`](https://huggingface.co/sapienzanlp/sentinel-src-mqm)   | Source text           | DA WMT17-20 + MQM WMT20-22  |
| [`sapienzanlp/sentinel-cand-da`](https://huggingface.co/sapienzanlp/sentinel-cand-da)   | Candidate translation | DA WMT17-20                 |
| [`sapienzanlp/sentinel-cand-mqm`](https://huggingface.co/sapienzanlp/sentinel-cand-mqm) | Candidate translation | DA WMT17-20 + MQM WMT20-22  |    
| [`sapienzanlp/sentinel-ref-da`](https://huggingface.co/sapienzanlp/sentinel-ref-da)     | Reference translation | DA WMT17-20                 |
| [`sapienzanlp/sentinel-ref-mqm`](https://huggingface.co/sapienzanlp/sentinel-ref-mqm)   | Reference translation | DA WMT17-20 + MQM WMT20-22  |

All metrics are based on XLM-RoBERTa large. All MQM sentinel metrics are further fine-tuned on MQM data starting from the DA-based sentinel metrics. All metrics can be found on [🤗 Hugging Face](https://huggingface.co/collections/sapienzanlp/mt-sentinel-metrics-66ab643b32aab06f3157e5c1).

## CLI

Except for `sentinel-metric-train`, all CLI commands included within this package require cloning and installing our fork of the [Google WMT Metrics evaluation repository](https://github.com/google-research/mt-metrics-eval). To do this, execute the following commands:

```bash
git clone https://github.com/prosho-97/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
```

Then, download the WMT data following the instructions in the **Downloading the data** section of the [README](https://github.com/prosho-97/mt-metrics-eval/blob/main/README.md).

### `sentinel-metric-score`
You can use the `sentinel-metric-score` command to score translations with our metrics. For example, to use a SENTINEL<sub>CAND</sub> metric:

```bash
echo -e 'Today, I consider myself the luckiest man on the face of the earth.\nI'"'"'m walking here! I'"'"'m walking here!' > sys1.txt
echo -e 'Today, I consider myself the lucky man\nI'"'"'m walking here.' > sys2.txt
sentinel-metric-score --sentinel-metric-model-name sapienzanlp/sentinel-cand-mqm -t sys1.txt sys2.txt
```

Output:

```
# input source sentences: 0     # input candidate translations: 4       # input reference translations: 0.

MT system: sys1.txt     Segment idx: 0  Metric segment score: 0.4837.
MT system: sys2.txt     Segment idx: 0  Metric segment score: 0.4722.
MT system: sys1.txt     Segment idx: 1  Metric segment score: 0.0965.
MT system: sys2.txt     Segment idx: 1  Metric segment score: 0.2735.

MT system: sys1.txt     Metric system score: 0.2901.
MT system: sys2.txt     Metric system score: 0.3729.
```

For a SENTINEL<sub>SRC</sub> metric instead:

```bash
echo -e "本文件按照 GB/T 1.1 一 202 久标准化工作导则第工部分:标准化文件的结构和起草规则的规定起惠。\n增加了本文件适用对象(见第 1 章)，" > src.txt
sentinel-metric-score --sentinel-metric-model-name sapienzanlp/sentinel-src-mqm -s src.txt
```

Output:

```
# input source sentences: 2     # input candidate translations: 0       # input reference translations: 0.

MT system: SOURCE       Segment idx: 0  Metric segment score: 0.1376.
MT system: SOURCE       Segment idx: 1  Metric segment score: 0.5106.

MT system: SOURCE       Metric system score: 0.3241.
```

You can also score data samples from the test sets of the WMT Metrics Shared Tasks. For example:

```bash
sentinel-metric-score --sentinel-metric-model-name sapienzanlp/sentinel-cand-mqm --batch-size 128 --testset-name wmt23 --lp zh-en --ref-to-use refA --include-human --include-outliers --include-ref-to-use --only-system --out-path data/metrics_results/metrics_outputs/wmt23/zh-en/SENTINEL_CAND_MQM 
```

Output:

```                                                                                                                                                                                                                    
lp = zh-en.                                                                                                                                                                                                         
# segs = 1976.                                                                                                                                                                                                      
# systems = 17.                                                                                                                                                                                                     
# metrics = 0.                                                                                                                                                                                                      
Std annotation type = mqm.                                                                                                                                                                                          
# refs = 2.                                                                                                                                                                                                         
std ref = refA.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                    
# MT systems to score in wmt23 for zh-en lp = 17.                                                                                                                                                                   
No domain is specified.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                    
# input source sentences: 1976  # input candidate translations: 33592   # input reference translations: 1976.

MT system: ONLINE-M     Metric system score: -0.0576.
MT system: ONLINE-Y     Metric system score: 0.0913.
MT system: NLLB_MBR_BLEU        Metric system score: 0.0725.
MT system: Yishu        Metric system score: 0.0703.
MT system: ONLINE-B     Metric system score: 0.066.
MT system: ONLINE-G     Metric system score: 0.0441.
MT system: refA Metric system score: -0.0193.
MT system: IOL_Research Metric system score: -0.0121.
MT system: ANVITA       Metric system score: 0.0603.
MT system: HW-TSC       Metric system score: 0.0866.
MT system: GPT4-5shot   Metric system score: 0.1905.
MT system: ONLINE-W     Metric system score: -0.0017.
MT system: synthetic_ref        Metric system score: 0.0667.
MT system: Lan-BridgeMT Metric system score: 0.1543.
MT system: ONLINE-A     Metric system score: 0.0231.
MT system: NLLB_Greedy  Metric system score: 0.0836.
MT system: ZengHuiMT    Metric system score: -0.0446.
```

`--out-path` points to the directory where the segment and system scores returned by the metric will be saved (`seg_scores.pickle` and `sys_scores.pickle`). Furthermore, you can provide the path to the model checkpoint using `--sentinel-metric-model-checkpoint-path` instead of specifying the Hugging Face model name with `--sentinel-metric-model-name`. Output scores can also be saved to a json file using the `--to-json` argument. Additionally, this command supports COMET metrics, which can be used with `--comet-metric-model-name` or `--comet-metric-model-checkpoint-path` argument.
 
For a complete description of the command (including also scoring csv data and limiting the evaluation to some specific WMT domain), you can use the `help` argument:

```bash
sentinel-metric-score --help
```

### `sentinel-metric-compute-wmt23-ranking`

The `sentinel-metric-compute-wmt23-ranking` command computes the WMT23 metrics ranking. For example, to compute the segment-level metrics ranking:

```bash
sentinel-metric-compute-wmt23-ranking --metrics-to-evaluate-info-filepath data/metrics_results/metrics_info/metrics_info_for_ranking.tsv --metrics-outputs-path data/metrics_results/metrics_outputs/wmt23 --k 0 --only-seg-level > data/metrics_results/metrics_rankings/seg_level_wmt23_final_ranking.txt
```

To group-by-item (Segment Grouping in the paper) when computing the segment-level Pearson correlation, use `--item-for-seg-level-pearson`. The output is located in [data/metrics_results/metrics_rankings/item_group_seg_level_wmt23_final_ranking.txt](data/metrics_results/metrics_rankings/item_group_seg_level_wmt23_final_ranking.txt). You also have the option to limit the segment-level ranking to using the Pearson correlation only, excluding the accuracy measure introduced by [Deutsch et al. (2023)](https://aclanthology.org/2023.emnlp-main.798/). To do this, use the `--only-pearson` flag. The output files will be located at [data/metrics_results/metrics_rankings/only_pearson_seg_level_wmt23_final_ranking.txt](data/metrics_results/metrics_rankings/only_pearson_seg_level_wmt23_final_ranking.txt) and [data/metrics_results/metrics_rankings/only_item_group_pearson_seg_level_wmt23_final_ranking.txt](data/metrics_results/metrics_rankings/only_item_group_pearson_seg_level_wmt23_final_ranking.txt).

You can add other MT metrics to this comparison by creating new folders in [data/metrics_results/metrics_outputs/wmt23](data/metrics_results/metrics_outputs/wmt23) for each language pair, containing their segment-level and system-level scores (check how the `seg_scores.pickle` and `sys_scores.pickle` files are created in [sentinel_metric/cli/score.py](sentinel_metric/cli/score.py)). To do this, you also have to include their info in the [data/metrics_results/metrics_info/metrics_info_for_ranking.tsv](data/metrics_results/metrics_info/metrics_info_for_ranking.tsv) file, specifying the metric name, the name of the folder containing its scores, and what gold references have been employed (or `src` if the metric is reference-free).

For a complete description of this command, execute:

```bash
sentinel-metric-compute-wmt23-ranking --help
```

### `sentinel-metric-compute-wmt-corrs`

The `sentinel-metric-compute-wmt-corrs` command can computes the metrics rankings on WMT for all possible combinations of correlation function and grouping strategy in a given language pair. For example, for zh-en language direction, in WMT23, you can use the following command:

```bash
sentinel-metric-compute-wmt-corrs --metrics-to-evaluate-info-filepath data/metrics_results/metrics_info/metrics_info_for_wmt_corrs.tsv --testset-name wmt23 --lp zh-en --ref-to-use refA --primary-metrics --k 0 > data/metrics_results/wmt_corrs/wmt23/zh-en.txt 
```

Similar to the previous command, you can include additional MT metrics by creating the necessary folders for the desired language pair and adding their info in the [data/metrics_results/metrics_info/metrics_info_for_wmt_corrs.tsv](data/metrics_results/metrics_info/metrics_info_for_wmt_corrs.tsv) file. For each new metric, you have to specify its name, whether it is reference-free, and the path to the folder containing its scores.

For a complete description of the command, execute:

```bash
sentinel-metric-compute-wmt-corrs --help
```

### `sentinel-metric-compute-corrs-matrix`

The `sentinel-metric-compute-corrs-matrix` command computes the correlations matrix for MT metrics in a given language pair, similar to the ones in the Appendix of our paper. To use it, two additional packages are required:

```bash
pip install matplotlib==3.9.1 seaborn==0.13.2
```

Then, considering zh-en language direction in WMT23 as an example, you can execute the following command:

```bash
sentinel-metric-compute-corrs-matrix --metrics-to-evaluate-info-filepath data/metrics_results/metrics_info/metrics_info_for_corrs_matrix.tsv --testset-name wmt23 --lp zh-en --ref-to-use refA --out-file data/metrics_results/corr_matrices/wmt23/zh-en.pdf
```

To specify which MT metrics to include in the correlations matrix, you can edit [data/metrics_results/metrics_info/metrics_info_for_corrs_matrix.tsv](data/metrics_results/metrics_info/metrics_info_for_corrs_matrix.tsv), specifying each metric's name, whether it is reference-free, and the path to its scores (`None` if already included in WMT).

For a complete description of this command, execute:

```bash
sentinel-metric-compute-corrs-matrix --help
```

### `sentinel-metric-train`

The `sentinel-metric-train` trains a new sentinel metric:

```bash
sentinel-metric-train --cfg configs/models/sentinel_regression_metric_model.yaml --wandb-logger-entity WANDB_ENTITY
```

Edit the files in the [configs](configs) directory to customize the training process. You can also start the training from a given model checkpoint (`--load-from-checkpoint`).

For a complete description of the command, execute:

```bash
sentinel-metric-train --help
```


## Sentinel metrics usage within Python:

```python
from sentinel_metric import download_model, load_from_checkpoint

model_path = download_model("sapienzanlp/sentinel-cand-mqm")
model = load_from_checkpoint(model_path)

data = [
    {"mt": "This is a candidate translation."},
    {"mt": "This is another candidate translation."}
]

output = model.predict(data, batch_size=8, gpus=1)
```

Output:
```python
# Segment scores
>>> output.scores
[0.347846657037735, 0.22583423554897308]

# System score
>>> output.system_score
0.28684044629335403
```

## Cite this work
This work has been published at [ACL 2024 (main conference)](https://2024.aclweb.org/program/main_conference_papers/). If you use any part, please consider citing our paper as follows:

```bibtex
@inproceedings{perrella-etal-2024-guardians,
    title = "Guardians of the Machine Translation Meta-Evaluation: Sentinel Metrics Fall In!",
    author = "Perrella, Stefano and
      Proietti, Lorenzo  and
      Scirè, Alessandro and
      Barba, Edoardo and
      Navigli, Roberto",
        booktitle = "Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL 2024)",
    year      = "2024",
    address   = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
}
```

## License
This work is licensed under [Creative Commons Attribution-ShareAlike-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
