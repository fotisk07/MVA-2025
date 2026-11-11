# Distributed training

Course: https://training-large-models-course.github.io/

- `Week1/`–`Week3/`: available lab notebooks and lecture slides.
- `project/`: Transformer training experiments, data preparation scripts, reports,
  and poster. See [project instructions](project/README.md).

## Consolidation notes

This directory contains ordinary files, not nested Git repositories.

The old `HW` entry was only a broken submodule pointer to the unavailable
`fotisk07/DisTrain-HW` repository. It was removed during consolidation with the
owner's approval; its homework contents could not be recovered. The recorded
commit was `246703c3431230aece9135002993374449748f18` (the standalone course repo
also referenced `fffdc82a84b353d0becd8ad26de9c41fc0c30375`).

The standalone project also referenced upstream
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291).
It is retained as a reference link rather than another submodule. The project's
model implementation is already included in `project/src/ssprox/model.py`;
no direct imports from the nanoGPT submodule were found.
