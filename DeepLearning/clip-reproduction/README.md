# CLIP Reproduction

Small reproduction project for CLIP-style training and evaluation in PyTorch.

## Setup

This repo uses uv. If you don't have uv installed please us a virtual environment. Otherwise you can just run
```bash
uv sync
```

## Train

Train with the default config (`conf/train_conf.yaml`):

```bash
uv run  scripts/train.py
```

Example overrides:

```bash
uv run scripts/train.py model=clip dataset=cifar100
uv run scripts/train.py model=resnet50_finetuning dataset=mnist
```

## Evaluate

Linear probe evaluation (`conf/eval_linear_probe_conf.yaml`):

```bash
uv run scripts/eval_linear_probe.py
```

Zero-shot OpenAI CLIP baseline:

```bash
uv run scripts/eval_openai_clip.py dataset=cifar100 image_size=224
```

The only supported image_size for OpenAI CLIP is `224`

## Main Files

- `scripts/train.py`: training loop for CLIP and classifier baselines.
- `scripts/eval_linear_probe.py`: feature extraction + logistic regression probe.
- `scripts/eval_openai_clip.py`: zero-shot evaluation with Hugging Face CLIP.
- `conf/train_conf.yaml`: training configuration.
- `conf/eval_linear_probe_conf.yaml`: probing configuration.
