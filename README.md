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

This work requires python 3.10 or higher. We recommend creating a new [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment to run the code. Execute the following commands to set it up:

```bash
conda create -n guardians-mt-eval python=3.10
conda activate guardians-mt-eval
pip install --upgrade pip
pip install -e .
```

## Data

The sentinel metric models are trained using Direct Assessments (DA) and Multidimensional Quality Metrics (MQM) annotations downloaded from the official [COMET repository](https://github.com/Unbabel/COMET/tree/master/data).

## Models

The following is a comprehensive list of the available sentinel metric models, including their inputs and the data used for training (z-scores):

| HF Model Name                                                                           | Input                 | Training Data |
|-----------------------------------------------------------------------------------------|-----------------------|---------------|
| [`sapienzanlp/sentinel-src-da`](https://huggingface.co/sapienzanlp/sentinel-src-da)     | Source text           | DA WMT17-20   |
| [`sapienzanlp/sentinel-src-mqm`](https://huggingface.co/sapienzanlp/sentinel-src-mqm)   | Source text           | MQM WMT20-22  |
| [`sapienzanlp/sentinel-cand-da`](https://huggingface.co/sapienzanlp/sentinel-cand-da)   | Candidate translation | DA WMT17-20   |
| [`sapienzanlp/sentinel-cand-mqm`](https://huggingface.co/sapienzanlp/sentinel-cand-mqm) | Candidate translation | MQM WMT20-22  |
| [`sapienzanlp/sentinel-ref-da`](https://huggingface.co/sapienzanlp/sentinel-ref-da)     | Reference translation | DA WMT17-20   |
| [`sapienzanlp/sentinel-ref-mqm`](https://huggingface.co/sapienzanlp/sentinel-ref-mqm)   | Reference translation | MQM WMT20-22  |

All metric models employ XLM-RoBERTa large as their backbone PLM, and all MQM sentinel metrics are trained starting from their DA counterpart model checkpoint. The models can be found on [🤗 Hugging Face](https://huggingface.co/collections/sapienzanlp/mt-sentinel-metrics-66ab643b32aab06f3157e5c1).

## CLI

Except for `sentinel-metric-train`, all commands included with this package require cloning and installing our fork of the official Google WMT Metrics evaluation [repository](https://github.com/google-research/mt-metrics-eval). To do this, execute the following commands:

```bash
git clone https://github.com/prosho-97/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
```

Then, download the WMT data following the instructions in the **Downloading the data** section of the [README](https://github.com/prosho-97/mt-metrics-eval/blob/main/README.md).

### `sentinel-metric-score`
You can use the `sentinel-metric-score` command to score sentences with our metrics. For example, to use a SENTINEL<sub>CAND</sub> metric:

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

You can also score samples coming from the official WMT Metrics test sets. For example:

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

Where `--out-path` defines the path to the directory where the segment and system scores returned by the metric will be saved (`seg_scores.pickle` and `sys_scores.pickle`). Alternatively, you can provide the path to the model checkpoint using `--sentinel-metric-model-checkpoint-path` instead of specifying the Hugging Face model name with `--sentinel-metric-model-name`. You can also save the output scores to a json file using the `--out-json` argument. Additionally, the command supports the use of COMET metrics, which can be specified using the `--comet-metric-model-name` (downloaded from Hugging Face) or `--comet-metric-model-checkpoint-path` argument. For a complete description of the command (including also scoring csv data, and specifying a WMT domain), run:

```bash
sentinel-metric-score --help
```

### `sentinel-metric-compute-wmt23-ranking`

The `sentinel-metric-compute-wmt23-ranking` command can be used to compute the WMT23 metrics ranking. For example, to compute the segment-level metrics ranking:

```bash
sentinel-metric-compute-wmt23-ranking --metrics-to-evaluate-info-filepath data/metrics_results/metrics_info/metrics_info_for_ranking.tsv --metrics-outputs-path data/metrics_results/metrics_outputs/wmt23 --k 0 --only-seg-level > data/metrics_results/metrics_rankings/seg_level_wmt23_final_ranking.txt
```

To use the item-grouping strategy for the segment-level Pearson correlation, as described in the paper, you only need to add the `--item-for-seg-level-pearson` flag. The output is currently located in [data/metrics_results/metrics_rankings/item_group_seg_level_wmt23_final_ranking.txt](data/metrics_results/metrics_rankings/item_group_seg_level_wmt23_final_ranking.txt). In both cases, you have the option to limit the segment-level ranking exclusively to Pearson correlation, excluding the Kendall correlation introduced by [Deutsch et al. (2023)](https://aclanthology.org/2023.emnlp-main.798/). To do this, simply add the `--only-pearson` flag. The output files are currently located at [data/metrics_results/metrics_rankings/only_pearson_seg_level_wmt23_final_ranking.txt](data/metrics_results/metrics_rankings/only_pearson_seg_level_wmt23_final_ranking.txt) and [data/metrics_results/metrics_rankings/only_item_group_pearson_seg_level_wmt23_final_ranking.txt](data/metrics_results/metrics_rankings/only_item_group_pearson_seg_level_wmt23_final_ranking.txt).

You can add other MT metrics to this comparison, creating new folders in [data/metrics_results/metrics_outputs/wmt23](data/metrics_results/metrics_outputs/wmt23) for each language pair, containing their segment-level and system-level scores (check how the `seg_scores.pickle` and `sys_scores.pickle` files are created in [sentinel_metric/cli/score.py](sentinel_metric/cli/score.py)). Finally, you have to include their info in the [data/metrics_results/metrics_info/metrics_info_for_ranking.tsv](data/metrics_results/metrics_info/metrics_info_for_ranking.tsv) file, specifying the metric name, the name of the folder containing the scores, and the references employed (or `src` if reference-less).

For a complete description of the command, run:

```bash
sentinel-metric-compute-wmt23-ranking --help
```

### `sentinel-metric-compute-wmt-corrs`

The `sentinel-metric-compute-wmt-corrs` command can be used to compute the metrics rankings on WMT for all possible combinations of `(correlation function, grouping strategy)` in a given language pair. For example, for zh-en in WMT23:

```bash
sentinel-metric-compute-wmt-corrs --metrics-to-evaluate-info-filepath data/metrics_results/metrics_info/metrics_info_for_wmt_corrs.tsv --testset-name wmt23 --lp zh-en --ref-to-use refA --primary-metrics --k 0 > data/metrics_results/wmt_corrs/wmt23/zh-en.txt 
```

Similar to the previous command, you can include additional MT metrics in this comparison by creating the necessary folders for the desired language pair and adding their details to the [data/metrics_results/metrics_info/metrics_info_for_wmt_corrs.tsv](data/metrics_results/metrics_info/metrics_info_for_wmt_corrs.tsv) file. For each new metric, specify its name, whether it is a QE metric, and the path to the folder containing its scores.

For a complete description of the command, run:

```bash
sentinel-metric-compute-wmt-corrs --help
```

### `sentinel-metric-compute-corrs-matrix`

The `sentinel-metric-compute-corrs-matrix` command can be used to compute the correlations matrix for MT metrics in a given language pair, similar to the ones shown in our paper. To use it, it is required to install two additional packages:

```bash
pip install matplotlib==3.9.1 seaborn==0.13.2
```

Then, for example, considering zh-en in WMT23:

```bash
sentinel-metric-compute-corrs-matrix --metrics-to-evaluate-info-filepath data/metrics_results/metrics_info/metrics_info_for_corrs_matrix.tsv --testset-name wmt23 --lp zh-en --ref-to-use refA --out-file data/metrics_results/corr_matrices/wmt23/zh-en.pdf
```

To decide which MT metrics include in the correlations matrix, edit the [data/metrics_results/metrics_info/metrics_info_for_corrs_matrix.tsv](data/metrics_results/metrics_info/metrics_info_for_corrs_matrix.tsv) file, specifying for each metric its name, whether it is QE, and the path to its scores (`None` if already included in WMT).

For a complete description of the command, run:

```bash
sentinel-metric-compute-corrs-matrix --help
```

### `sentinel-metric-train`

The `sentinel-metric-train` command can be used to train a new sentinel metric model:

```bash
sentinel-metric-train --cfg configs/models/sentinel_regression_metric_model.yaml --wandb-logger-entity WANDB_ENTITY
```

Edit the files in the [configs](configs) directory to customize the training process. You can also start the training from a given model checkpoint (`--load-from-checkpoint`).

For a complete description of the command, run:

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
