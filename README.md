# Collaborative Threshold Watermarking

Official implementation of [Collaborative Threshold Watermarking](https://arxiv.org/abs/2602.10765) (arXiv 2602.10765).

## Overview

We propose *(t,K)-threshold watermarking* for federated learning. K clients collaboratively embed a shared watermark during training. Only coalitions of at least t clients can reconstruct the watermark key and verify a suspect model. Secret sharing prevents verification by smaller coalitions and protects the key during verification.

The watermark remains detectable at scale (K=128) with minimal accuracy degradation and resists fine-tuning attacks with up to 20% of training data.

## Setup

```bash
conda create -n fedwm python=3.11 -y
conda activate fedwm
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Usage

**Baseline (no watermark):**
```bash
python example/baseline_resnet18.py          # CIFAR-10
python example/baseline_resnet18_cifar100.py # CIFAR-100
python example/baseline_tinyimagenet.py      # TinyImageNet
```

**Trusted server:**
```bash
python example/trusted_server_resnet18.py
python example/trusted_server_resnet18_cifar100.py
python example/trusted_server_tinyimagenet.py
```

**Untrusted server:**
```bash
python example/untrusted_server_resnet18.py
python example/untrusted_server_resnet18_cifar100.py
python example/untrusted_server_tinyimagenet.py
```

**Trustless key generation:**
```bash
python trustless_keygen/simulation_distributed.py
python trustless_keygen/benchmark_distributed.py
```

## Project Structure

```
src/                  # Core library
  Client.py           # Base federated client
  ClientResNet18.py   # ResNet18 client (CIFAR-10/100)
  ResNet.py           # ResNet architecture
  data_utils.py       # Dataset utilities
  utils.py            # Watermark generation and verification
example/              # Experiment scripts
trustless_keygen/     # MPC-based distributed key generation
```

## Citation

```bibtex
@article{bakr2026collaborative,
  title   = {Collaborative Threshold Watermarking},
  author  = {Bakr, Tameem and Ambreth, Anish and Lukas, Nils},
  journal = {arXiv preprint arXiv:2602.10765},
  year    = {2026}
}
```
