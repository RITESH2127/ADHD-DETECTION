# Contributing

Contributions should improve correctness, reproducibility, documentation or research transparency.

## Guidelines

- Keep raw EEG/MRI recordings and participant-identifying information out of Git.
- Do not silently change preprocessing, labels, participant splits or evaluation definitions.
- Keep dataset paths configurable rather than machine-specific.
- Distinguish the main EEG-only experiment from the separate exploratory EEG–MRI notebook.
- Do not describe exploratory metrics as clinical evidence or validated biomarkers.
- Do not repeatedly tune against the held-out test set.

## Experiment reports

Document the research question, notebook revision, dataset source and version, participant counts per split, preprocessing settings, model configuration, selection rule, metrics at both window and participant level, uncertainty estimates and known limitations. Do not assign an ablation label unless the changed component is recorded explicitly.

## Validation

Before submitting a change, check that notebooks remain valid JSON, CSV files remain readable and documentation links resolve. For model changes, also run the relevant notebook cells in a documented environment and state clearly what was and was not executed. A structure check is not proof of model correctness.
