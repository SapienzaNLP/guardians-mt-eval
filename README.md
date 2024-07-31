<div align="center">

# Guardians of the Machine Translation Meta-Evaluation: Sentinel Metrics Fall In!

[![Conference](https://img.shields.io/badge/ACL-2024-4b44ce
)](https://2024.aclweb.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://2024.aclweb.org/program/main_conference_papers/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

</div>

## About the Project

This is the official repository of the ACL 2024 paper [*Guardians of the Machine Translation Meta-Evaluation: Sentinel Metrics Fall In!*](https://2024.aclweb.org/program/main_conference_papers/).

## Abstract

Annually, at the Conference of Machine Translation (WMT), the Metrics Shared Task organizers conduct the Machine Translation (MT) meta-evaluation, assessing MT metrics' capabilities according to their correlation with human judgments. Their results guide researchers toward enhancing the next generation of metrics and MT systems.
With the recent introduction of neural metrics, the field has witnessed notable advancements. Nevertheless, the inherent opacity of these metrics has posed substantial challenges to the meta-evaluation process.
This work highlights two issues with the meta-evaluation framework currently employed in WMT, and assesses their impact on the metrics rankings. To do this, we introduce the concept of sentinel metrics, which are designed explicitly to scrutinize the meta-evaluation process's accuracy, robustness, and fairness. By employing sentinel metrics, we aim to validate our findings, and shed light on and monitor the potential biases or inconsistencies in the rankings. We discover that the present meta-evaluation framework favors two categories of metrics: i) those explicitly trained to mimic human quality assessments, and ii) continuous metrics. Finally, we raise concerns regarding the evaluation capabilities of state-of-the-art metrics, emphasizing that they might be basing their assessments on spurious correlations found in their training data.

## Setup

This work requires python 3.9 or above. We suggest the creation of a new [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment to run the code. To do so, run the following commands:

```bash
conda create -n guardians-mt-eval python=3.9
conda activate guardians-mt-eval
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Data

The data used in this work, including Direct Assessments and Multidimensional Quality Metrics, can be downloaded [here](https://github.com/Unbabel/COMET/tree/master/data).

## Models

The following is a comprehensive list of all available sentinel regression metric models:

- [**SENTINEL<sub>CAND</sub>**](https://drive.google.com/file/d/1uSUecrACI_RApFiVz2XtIQJQ2dFf_uNN/view?usp=sharing): Sentinel regression metric model that takes in input only the candidate translation.
- [**SENTINEL<sub>SRC</sub>**](https://drive.google.com/file/d/1BRRk7VOW4ri0fkOlPjreHJDgahohBIhY/view?usp=sharing): Sentinel regression metric model that takes in input only the source sentence.
- [**SENTINEL<sub>REF</sub>**](https://drive.google.com/file/d/1PAgh5fxVtkMoA88jCmWEQxLX9JtVXe3l/view?usp=sharing): Sentinel regression metric model that takes in input only the reference translation.

## Scoring within Python:

```python
from sentinel_metric import download_model, load_from_checkpoint

model_path = # set to the checkpoint path of the model to use

# Load the model checkpoint:
model = load_from_checkpoint(model_path)

# Data must be in the following format:
data = [
    {
        "src": "10 到 15 分钟可以送到吗",
        "mt": "Can I receive my food in 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    },
    {
        "src": "Pode ser entregue dentro de 10 a 15 minutos?",
        "mt": "Can you send it for 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    }
]
# Call predict method:
model_output = model.predict(data, batch_size=8, gpus=1)
```

As output, we get the following information:
```python
# Sentence-level scores (list)
>>> model_output.scores
[0.9822099208831787, 0.9599897861480713]

# System-level score (float)
>>> model_output.system_score
0.971099853515625
```

## Cite this work
This work has been published at ACL 2024 (main conference). If you use any part, please consider citing our paper as follows:

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