# Data and Code for "Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks" (KDD 2024)

This repo contains the data and code for the following paper: 

Jiaying Wu, Jiafeng Guo, Bryan Hooi. Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks, ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) 2024. [![arXiv](https://img.shields.io/badge/arXiv-2310.10830-b31b1b.svg)](https://arxiv.org/abs/2310.10830)


## Abstract

It is commonly perceived that fake news and real news exhibit distinct writing styles, such as the use of sensationalist versus objective language. However, we emphasize that style-related features can also be exploited for style-based attacks. Notably, the advent of powerful Large Language Models (LLMs) has empowered malicious actors to mimic the style of trustworthy news sources, doing so swiftly, cost-effectively, and at scale. Our analysis reveals that LLM-camouflaged fake news content significantly undermines the effectiveness of state-of-the-art text-based detectors (up to 38% decrease in F1 Score), implying a severe vulnerability to stylistic variations. To address this, we introduce SheepDog, a style-robust fake news detector that prioritizes content over style in determining news veracity. SheepDog achieves this resilience through (1) LLM-empowered news reframings that inject style diversity into the training process by customizing articles to match different styles; (2) a style-agnostic training scheme that ensures consistent veracity predictions across style-diverse reframings; and (3) content-focused veracity attributions that distill content-centric guidelines from LLMs for debunking fake news, offering supplementary cues and potential interpretability that assist veracity prediction. Extensive experiments on three real-world benchmarks demonstrate SheepDog's style robustness and adaptability to various backbones.

# 🧠 RoBERTa Training and Evaluation Script Guide (PolitiFact Dataset)

This repository contains a script to evaluate fine-tuned **RoBERTa models** on adversarial variants of the **PolitiFact** dataset.

The evaluation computes **accuracy, precision, recall, F1-scores** (macro and per-class), and logs the results to both **terminal** and a **log file** for reproducibility.



## ⚙️ Environment Setup

This project uses a **Python 3.10.18 (CPython)** environment, preferably managed via **[UV](https://github.com/astral-sh/uv)** (a fast Python package and environment manager).

### Prerequisites

You need **UV** installed to use the commands below efficiently.

### Installation Guide using UV



```pip install uv```

```uv pip list```

```uv python list```

```uv python install cpython-3.10.18-windows-x86_64-none```

```uv python list```

```uv venv env --python cpython-3.10.18-windows-x86_64-none```

#if you have conda then first deactivate that
```conda deactivate```

```uv venv env --python cpython-3.10.18-windows-x86_64-none```


# Activate the environment
### For Windows:
source env/Scripts/activate
### For Linux/macOS:
source env/bin/activate
## Data
Our work is based on the `PolitiFact` and `GossipCop` datasets from the [FakeNewsNet benchmark](https://github.com/KaiDMML/FakeNewsNet), and the `LUN` dataset from [(Rashkin et al., 2017)](https://aclanthology.org/D17-1317.pdf). 

We provide the data files utilized for training and evaluating SheepDog under `data/`. In our data files, the label `0` represents real news, and the label `1` represents fake news.

**Original Unaltered Training / Test Articles**

The `.pkl` files under `data/news_articles/` contain the unaltered news article texts. Please refer to Section 6.1.1 of our paper for more details.

**Adversarial Test Sets**

The `.pkl` files under `data/adversarial_test/` contain the four adversarial test sets under LLM-empowered style attacks, denoted as A through D. Please refer to Section 4.1 and Table 4 of our paper for a detailed formulation.

**LLM-Empowered News Reframings**

The `.pkl` files under `data/reframings/` contain the style-diverse reframings of training articles. Please refer to Section 5.1 of our paper for detailed descriptions.  

**Content-Focused Veracity Attributions**

The `.pkl` files under `data/veracity_attributions/` contain the content-focused veracity attributions of training articles. Here, each article is assigned 4 binary labels according to the following 4 attributions: (1) lack of credible sources, (2) false or misleading information, (3) biased opinion, and (4) inconsistencies with reputable sources. Please refer to Section 5.3 of our paper for detailed descriptions. 


## Run SheepDog
 

Start training with the following command:

```
python src/sheepdog.py
```
Start evaluation with the following command:

```
python src/evaluate.py
```

Model checkpoints will be saved under `checkpoints/`, and results will be saved under `logs/`. 

Additionally, under `logs/logs_archive_all4_adv/`, we provide archived experiment logs for SheepDog on both the original test set and adversarial test sets A-D.

## Installation Guide for PyTorch with CUDA
- run nvidia-smi to see the CUDA version of the GPU, in my case it is 12.6
- PyTorch with CUDA : 12.6  must be installed via index-url
- Visit pytorch.org to get the correct command for your system configuration.
- So we will skip torch, torchvision, torchaudio here in requirements.txt you need to install this via terminal using uv pip command if required but mostly it will be installed when you install pythorch with CUDA, using the proper command listed in pytorch.org
