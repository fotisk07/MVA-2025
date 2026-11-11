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

## Remaining submodule review

`MVA-2025` is public. Publication approval covers only the imported DisTrain files,
not additional private submodule content.

- `fotisk07/DisTrain-project`: most files already match the consolidated project,
  but its `nanoGPT` submodule still needs an explicit handling decision.

The existing `DisTrain/HW` gitlink also needs review: the consolidated repository
has no root `.gitmodules`, and its referenced `DisTrain-HW` repository was not
present in the accessible repository inventory. Do not assume it is backed up.

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
