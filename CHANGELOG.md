# Change log

## 1.0.0 — 2026-07-28

- Converted `TDLI.ipynb` into `Utility/time_delay_interpolation.py`.
- Converted `TDPSPL.ipynb` into `Utility/time_delay_pspline.py`.
- Added single-system and checkpointed batch command-line workflows.
- Preserved exact original parameter adaptations in named JSON profiles.
- Preserved the original 25-system batch pair list as CSV.
- Added eight focused module-use notebooks plus a start notebook.
- Added Windows and macOS installation procedures.
- Added data, workflow, methodology and supervisor-handoff documentation.
- Made optional ML modules lazy so core scientific utilities do not require
  PyTorch or `sktime`.
- Added Apple Silicon MPS auto-detection for neural-network inference.
- Corrected `IdSep.build_component_dict` to accept both DataFrames and CSV paths.
- Archived original exploratory notebooks under `notebooks/legacy/`.
