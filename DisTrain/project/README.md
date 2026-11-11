# Transformer training experiments

A nanoGPT-based Transformer training project with Hydra configuration and
experiments on training stability. Reports and poster sources are in `rendus/`.

## Environment

From this directory:

```bash
uv sync
```

Training code is in `src/ssprox/`; configuration is in `src/ssprox/conf/`.
No nanoGPT submodule checkout is required by the included Python imports.

## Data and training

Prepare a dataset using a script in `data/`. For example:

```bash
uv run python data/shakespeare_char/prepare.py
uv run ssprox-train data_dir=data/shakespeare_char wandb_log=false
```

Dataset preparation downloads external data. Training can be expensive; review
model and training settings before running. The default configuration uses
`data/openweb`, a GPT-2-small model, and online Weights & Biases logging. Override
`wandb_log=false` to disable that logging, as shown above.

Alternate experiment configurations:

```bash
uv run ssprox-train --config-name claim1 wandb_log=false
uv run ssprox-train --config-name claim2 wandb_log=false
```

These commands document the existing entry points; the training runs have not
been executed as part of repository consolidation.

## Upstream reference

[karpathy/nanoGPT at 3adf61e](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291).
The local model implementation is `src/ssprox/model.py`.
