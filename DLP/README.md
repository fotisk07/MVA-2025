# Deep Learning in Practice

Lab notebooks and assignments for the MVA Deep Learning in Practice course.

Course page: <https://www.lri.fr/~gcharpia/dlip/>

## Contents

- `Week1/`: multiclass classification, regression, and PyTorch basics.
- `Week2/`: Grad-CAM.
- `Week3/`: graph neural networks and message passing.
- `Week4/`: Grad-CAM and last-layer methods.
- `Week5/`: practical assignment and saved results.
- `Week6/`: generative models.

## Setup

This folder uses `uv`:

```bash
uv sync
```

Start Jupyter locally with:

```bash
uv run --with jupyter jupyter notebook
```

For cluster notebooks, use `jupyter_job.batch` or the week-specific Slurm files when present.
