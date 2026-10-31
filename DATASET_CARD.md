# Dataset Card — EEG Research Data

## Summary

The main experiment processes subject-level EEG recordings supplied as MATLAB `.mat` files. Raw recordings are not committed to this repository. Users must obtain the source data through its authorized distribution channel.

This card does not assert a dataset name, license, recruitment protocol, demographic profile or clinical label definition that has not been verified from authoritative dataset documentation.

## What is known from the implementation

| Attribute | Repository evidence |
|---|---|
| Modality | EEG |
| Expected file format | MATLAB `.mat` |
| Expected channels | 19 EEG channels |
| Sampling frequency | 128 Hz |
| Windowing | 256 samples (2 seconds) |
| Main task labels | ADHD and Control |
| Main notebook split | Participant-grouped train, validation and test partitions |
| Raw data distribution | Not included in this Git repository |
| Data path | Configured in the main notebook; originally points to a mounted input directory |

## Before using the data

Researchers should independently verify and document the dataset title and publisher, canonical source, version and download date, license and access conditions, ethics and consent requirements, participant inclusion criteria, label provenance, demographics, class balance, missing data, EEG device, acquisition protocol, montage, recording environment and repeated sessions.

Do not infer these details from filenames or model code.

## Privacy and handling

- Do not commit raw EEG, MRI, participant identifiers, consent documents or restricted metadata.
- Keep data outside the repository and configure local paths in the notebook.
- Follow the source dataset's access terms and institutional requirements.
- Avoid publishing participant-level predictions or visualizations that could disclose sensitive information.

## Known limitations

The repository does not include enough verified source documentation to provide a complete dataset datasheet. Dataset provenance and licensing must be completed by the researcher before redistribution or publication. The MIT license in this code repository does not grant rights to the underlying data.

## Reproducibility checklist

For every experiment, record the dataset source and version, file manifest, participant counts by split and class, preprocessing configuration, excluded recordings/windows, and a cryptographic hash or version identifier for the dataset manifest where permitted.
