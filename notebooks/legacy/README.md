# Legacy research notebooks

These notebooks are preserved as the historical record of the exploratory
work. They may contain hard-coded IDs, paths, long-running cells and duplicated
function definitions.

Use the numbered notebooks in the parent directory for normal work. In
particular:

- `TDLI.ipynb` was converted to `Utility/time_delay_interpolation.py`;
- `TDPSPL.ipynb` was converted to `Utility/time_delay_pspline.py`;
- their exact parameter adaptations are preserved under the
  `legacy_notebook` profiles in `configs/time_delay_profiles.json`.

Do not edit a legacy notebook to change a production algorithm. Update the
corresponding Python module, test it, then update its guided notebook.
