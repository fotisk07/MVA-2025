# MVA consolidation

## Imported snapshots

Public-source imports and five missing DisTrain lab / lecture files are included.
The owner explicitly approved publishing these DisTrain files from their private source.

| Source repository | Destination | Scope |
| --- | --- | --- |
| `fotisk07/AlteGrad-Challenge` | `AlteGrad/challenge/` | Challenge snapshot |
| `fotisk07/DisTrain` (private) | `DisTrain/Week1/`–`Week3/` | Five lab notebooks / lecture PDFs; publication approved |
| `fotisk07/DLMI` | `DLMI/` | Missing `Guidelines.pdf` |
| `fotisk07/TimeSeries-MiniProject` | `TimeSeries/project/` | Missing paper, presentation PDF, and report PDF |

Source commits and checksums of added files are recorded in `CONSOLIDATION.json`.
Generated Python caches and selected LaTeX/log artifacts were omitted. Submodule
configuration was not copied because nested projects are being consolidated into
ordinary directories. Existing files were not overwritten. Source Git histories,
issues, releases, and branches have not been imported or deleted.

## DisTrain submodules resolved

The owner approved best-effort cleanup even if some old homework cannot be recovered.
The broken `DisTrain/HW` gitlink was removed: its source repository is unavailable,
and no homework contents are claimed to be backed up. Commit references are retained
in `DisTrain/README.md` in case a copy is found later.

The project uses its included model implementation. The old upstream nanoGPT
submodule is documented as a pinned reference link, not copied or required as a
nested checkout. Python caches and empty training logs were removed from this
project. Original standalone repositories remain unchanged.

## Excluded from MVA

RL and VIC are not MVA courses (confirmed by the owner). Their local imports were
removed from this branch and preserved outside the repository in
`~/10_Projects/mva-excluded-backup/`. Original GitHub repositories are unchanged.

## Needs classification

- `Machine-Learning-Challenge`: Beacon Biosignals sleep staging challenge; confirm
  whether this belongs to MVA or Centrale before choosing a destination.
- `M2AD`: BCG climate/barley consulting case study; left out of MVA.

## Before retiring standalone repositories

1. Review and approve private-source handling.
2. Check submodules, relative paths, environments, and representative entry points.
3. Review imported notebooks, datasets, and outputs for sensitive material.
4. Push and verify the consolidated repository.
5. Preserve original repositories/history; archive or change visibility only with
   explicit approval. Archiving alone does not hide repositories.
