# MVA 2025

Coursework, labs, projects, reports, and challenge submissions for the MVA 2025-2026 master.

Each course or project is kept in its own folder. Most runnable projects use `uv` and include their own `pyproject.toml`, `uv.lock`, or local `README.md`.

## Repository Map

- `AlteGrad/`: labs and the molecular graph/text challenge in `challenge/`. Course: https://moodle.ip-paris.fr/course/info.php?id=10507
- `DeepLearning/`:
- `DLMI/`:
- `DLP/`: https://www.lri.fr/~gcharpia/dlip/
- `DisTrain/`: https://training-large-models-course.github.io/
- `GDA/`: https://www.jeanfeydy.com/Teaching/index.html
- `Interactions/`: interacting particle systems / systemic risk experiments.
- `RepLer/`: https://pietrogori.github.io/teaching/RepLearnMVA
- `TimeSeries/`: (https://www.laurentoudre.fr/ast.html)

## Consolidation

See [CONSOLIDATION.md](CONSOLIDATION.md) for source repositories, pending imports, and privacy limitations. `CONSOLIDATION.json` records imported source commits and file checksums. Imports are file snapshots; original Git histories remain in the standalone repositories.

## Setup

There is no single environment for the whole repository. Enter the folder you want to work in and follow its local README when available.

For `uv` projects:

```bash
cd <project-folder>
uv sync
```

Run scripts with:

```bash
uv run python <script.py>
```

Start notebooks with:

```bash
uv run --with jupyter jupyter notebook
```

Some folders also provide Slurm helpers such as `run_slurm.sh` or `jupyter_job.batch` for GPU cluster runs.
