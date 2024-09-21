from setuptools import setup

setup(
    name="sentinel_metric",
    version="1.0",
    author="Lorenzo Proietti, Stefano Perrella",
    author_email="lproietti@diag.uniroma1.it, perrella@diag.uniroma1.it",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SapienzaNLP/guardians-mt-eval",
    packages=["sentinel_metric"],
    install_requires=[
        "numpy>=1.24.4",
        "pandas>=2.1.3",
        "scipy>=1.11.3",
        "transformers>=4.35.0",
        "tokenizers>=0.14.1",
        "sentencepiece>=0.1.99",
        "torch>=2.1.0",
        "pytorch-lightning>=2.1.1",
        "torchmetrics>=0.10.3",
        "wandb>=0.17.5",
        "rich>=13.7.1",
        "unbabel-comet>=2.2.0",
    ],
    python_requires=">=3.10.0",
    entry_points={
        "console_scripts": [
            "sentinel-metric-score=sentinel_metric.cli.score:score_command",
            "sentinel-metric-compute-wmt23-ranking=sentinel_metric.cli.compute_final_wmt23_ranking:compute_final_wmt_ranking_command",
            "sentinel-metric-compute-corrs-matrix=sentinel_metric.cli.compute_correlations_between_metrics:compute_correlations_between_metrics_command",
            "sentinel-metric-compute-wmt-corrs=sentinel_metric.cli.compute_correlations_on_wmt:compute_wmt_corrs_command",
            "sentinel-metric-train=sentinel_metric.cli.train:train_command",
        ]
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
        "Programming Language :: Python :: 3.10",  # Specific to Python 3.10 and higher
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
