# Supervisor handoff

## What is immediately runnable

- preprocessing, grouping, plotting and both time-delay methods with
  `requirements-core.txt`;
- quasar and lens inference with the complete `requirements.txt`;
- all guided notebooks after supplying a canonical CSV in `data/`;
- batch P-spline processing with the supplied pair table.

## What is preserved

- every trained model from the supplied archive;
- prior batch time-delay results;
- the original notebooks under `notebooks/legacy/`;
- the exact time-delay settings from both original notebooks under the
  `legacy_notebook` profiles;
- the original list of 25 batch component pairs.

## Recommended review sequence

1. `README.md`
2. `docs/INSTALLATION.md`
3. `notebooks/00_start_here.ipynb`
4. `notebooks/02_preprocessing.ipynb`
5. classification notebooks 04–06
6. time-delay notebooks 07–08
7. `docs/TIME_DELAY_METHODS.md`

## Scientific cautions

- Classifier probabilities rank candidates; they do not confirm lensing.
- A large Gaia identifier must be handled as a string or 64-bit integer.
- A minimum objective at the edge of the delay range requires rerunning with a
  wider range.
- Low overlap can produce attractive but poorly supported delays.
- Multimodal Monte Carlo samples must be shown, not hidden behind one standard
  deviation.
- Applying the trained models to non-Gaia data requires new validation.

## Maintenance rules

- Add reusable algorithms to `Utility/*.py`, not directly to a result notebook.
- Add or update the corresponding guided notebook whenever a public function
  or hyperparameter changes.
- Store stable parameter adaptations in `configs/time_delay_profiles.json`.
- Keep raw datasets out of the source archive unless redistribution is
  authorised.
- Never overwrite trained model files without recording provenance and
  validation metrics.
- Verify model integrity after transfer with `sha256sum -c SHA256SUMS.txt`
  from the `models/` directory (or `Get-FileHash` on Windows).
