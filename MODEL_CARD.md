# Model Card — EEG Hybrid CNN–ViT Research Prototype

## Model overview

The principal experiment combines a spectral-temporal branch (in-graph STFT, CNN and Vision Transformer blocks) with a functional-connectivity branch (171 pairwise Pearson correlations from 19 EEG channels). A late-fusion classifier produces Control and ADHD class scores.

The primary implementation is [sensitivitytestadhdcode.ipynb](sensitivitytestadhdcode.ipynb). This card describes that EEG-only experiment, not the separate exploratory EEG–MRI notebook.

## Intended use

- Educational study of EEG preprocessing and deep-learning pipelines.
- Reproduction and critique of the experiments stored in this repository.
- Research prototyping under an appropriately approved data-use protocol.

## Out-of-scope use

- Clinical diagnosis, screening, triage or treatment decisions.
- Use as a medical device or substitute for a qualified clinician.
- Employment, education, insurance or other consequential decisions about individuals.
- Claims that model-ranked channel pairs are validated ADHD biomarkers.

**This model is not clinically validated.** Its outputs must not be used to make decisions about a person's health.

## Inputs and outputs

| Item | Description |
|---|---|
| Input window | 256 samples × 19 EEG channels |
| Sampling frequency | 128 Hz |
| Window duration | 2 seconds |
| Spectral branch | STFT magnitude representation followed by CNN and ViT |
| Connectivity branch | 171 unique pairwise channel-correlation values |
| Output | Softmax scores for Control and ADHD classes |

Different EEG systems, montages, sampling rates or file structures require explicit validation and may require adaptation.

## Reported evaluation

The saved baseline report, [final_evaluation_metrics_base.txt](final_evaluation_metrics_base.txt), reports a window-level AUROC of 0.9261 across 1,000 windows, participant-level AUROC of 1.0000 across 18 held-out participants (9 per class), and window-level accuracy of 85.30%.

These are historical results from committed experiment artefacts, not independently reproduced results or a guarantee of future performance. The participant-level cohort is very small; a perfect point estimate on 18 people is statistically fragile. Multiple saved experiments also report differing results on the same cohort.

## Data and evaluation considerations

- Raw participant recordings are not included in this repository.
- The main notebook uses participant-grouped train/validation/test partitions and checks for participant overlap.
- Windows from one participant are correlated and should not be treated as independent people.
- The saved reports do not demonstrate external, multi-site validation.
- Repeated inspection of the held-out cohort across experiment variants can introduce model-selection bias.
- The implemented subject-specific normalization is retrospective and should not be assumed suitable for prospective inference.

## Known limitations and risks

- Small held-out participant count.
- No demonstrated external validation across sites, devices or populations.
- Potential dataset- and acquisition-specific effects.
- Calibration and subgroup performance require further study.
- Grad-CAM and connectivity rankings are descriptive model-analysis outputs, not causal explanations.
- The separate EEG–MRI prototype is exploratory and is not evidence of a validated multimodal model.

## Recommendations before further research use

1. Verify dataset provenance, permissions, labels and participant-level metadata.
2. Freeze a configuration before evaluating on an untouched external cohort.
3. Use nested/repeated participant-grouped validation for model selection.
4. Report participant counts, class composition, uncertainty and calibration alongside window metrics.
5. Evaluate subgroup performance, robustness to artifacts and dataset shift.
6. Define an inference-time normalization strategy that does not depend on unavailable future recording data.
7. Have domain experts review any proposed clinical interpretation.

## Versioning

The main notebook is the source of truth for implementation details. When reporting results, record the Git commit, dataset version, preprocessing settings, split seed, package environment and exact notebook configuration.
