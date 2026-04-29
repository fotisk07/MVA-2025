# Representation Learning

Course material and lab notebooks for the MVA 2025 [Representation Learning](https://pietrogori.github.io/teaching/RepLearnMVA) class

**Objectives and Topics**: Good and expressive data representations can improve the accuracy of machine learning problems and ease interpretability and transfer. For computer vision and medical imaging tasks, handcrafting good data representations, a.k.a. feature engineering, was traditionally hard. Deep Learning has changed this paradigm by allowing the automatic discovery of good representations from data. This is known as representation learning. The objective of this course is to provide an introduction to representation learning in computer vision and medical imaging applications. We will cover the following topics:

## Contents

- `Week1/`: PyTorch introduction and adversarial examples
- `Week2/`: Domain adaptation
- `Week3/`: Self-supervised learning with rotations
- `Week4/`: Contrastive self-supervised learning
- `Week5/`: Vision Transformers
- `Week6/`: Masked autoencoders
- `Week7/`: VAEs and MaskGIT
- `Week8/`: Model interpretability on MedMNIST

## Setup

This folder uses `uv` for dependency management.

```bash
uv sync
```

To start Jupyter locally:

```bash
uv run --with jupyter jupyter notebook
```

## Slurm

Submit a command on a GPU partition:

```bash
./run_slurm.sh prod10 uv run python script.py
```

Available partitions are `prod10`, `prod20`, `prod40`, and `prod80`.

To start a Jupyter notebook on the cluster:

```bash
sbatch jupyter_job.batch
```

Then check the generated log in `logs/` for the SSH tunnel command.
