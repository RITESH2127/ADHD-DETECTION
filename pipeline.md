```
[1. Determinism & Environment Initialization]
   ├── Set OS environment flags (PYTHONHASHSEED=42, XLA flags)
   ├── Set global random seeds (tf.keras.utils.set_random_seed(42), op determinism)
   └── Initialize directory structure (logs, models, preproc/train/eval plots)

[2. Centralized Hyperparameters (`Config`)]
   ├── Sampling Frequency (SFREQ=128 Hz), Window Size (256 samples / 2s)
   ├── Channel Configuration (19-channel standard 10-20 EEG)
   ├── Preprocessing Bounds (1.0–40.0 Hz bandpass, ICA=15 components, PTP threshold=150 µV)
   ├── Architecture Config (D_MODEL=64, VIT_BLOCKS=3, VIT_HEADS=4)
   └── Training Hyperparameters (Batch size=64, Epochs=50/60, AdamW, Focal Loss)

[3. Clinical Data Ingestion, Hashed Caching & Preprocessing]
   ├── Generate MD5 hash of preprocessing signature (hyperparameters + ICA seed)
   ├── [CACHE CHECK] Load verified `.pkl` payload if signature matches else run MNE pipeline:
   │    ├── Rescaling: Load `.mat` files -> Scale voltages (`1e-7`)
   │    ├── Channel renaming: ('T3'->'T7', 'T4'->'T8', 'T5'->'P7', 'T6'->'P8') & 10-20 montage setup
   │    ├── Flatline channel detection & interpolation: (`BAD_THRESHOLD = 1e-10`)
   │    ├── Bandpass filtering & CAR: 1.0–40.0 Hz FIR bandpass filtering & Common Average Reference (CAR)
   │    ├── MNE Infomax ICA:  (`n_components=15`) + ICLabel classification
   │    ├── Artifact exclusion: (`prob > 0.70` for blink, heart, muscle)
   │    └── Patient-wise Z-score normalization: with telemetry fallback
   └── write to cache (atomic) + ID collision & subject count assertions (`== 121`)

[4. Subject-Isolated Splitting & Windowing]
   ├── `StratifiedGroupKFold(n_splits=7)` (Train: 5 folds, Val: 1 fold, Test: 1 fold) -> Zero patient overlap
   ├── Sliding window extraction with overlap (`TRAIN_OVERLAP=0.85`, `TEST_OVERLAP=0.0`)
   ├── Absolute amplitude window rejection (`> 150 µV`)
   ├── Vanishing subject verification assertion (`len(vanished) == 0`)
   └── Categorical label encoding, class balance calculation, normalized Focal Alpha computation, global train shuffle

[5. Asynchronous `tf.data` Pipeline & Connectivity Extraction]
   ├── Curriculum mixup callback & resource logging callback hooks (currently disabled)
   ├── Batch-wise Pearson correlation connectivity calculation (upper triangle extraction -> 171 features)
   └── `tf.numpy_function` wrappers mapped to `tf.data.Dataset` (`prefetch(AUTOTUNE)`, `deterministic=True`)

[6. IEEE Hybrid Neural Architecture (`GPU_STFT_Layer` -> CNN -> ViT -> Pearson)]
   ├── STFT and decibel scaling: Raw EEG Input (`256 × 19`) ──> `GPU_STFT_Layer` (128 frame, 8 step, 127 FFT, 41 freq bins up to 40 Hz, log-dB)
   ├── CNN Spatial-Temporal Blocks:
   │    ├── Block 1: Conv2D (1,7) + Conv2D (7,1) + MaxPool2D(2,2) + SpatialDropout2D(0.3)
   │    ├── Block 2: Conv2D(3,3) + MaxPool2D(2,1) + SpatialDropout2D(0.3)
   │    └── Block 3: Conv2D(3,3) + MaxPool2D(2,1) + SpatialDropout2D(0.4)
   ├── Patch Projection Conv2D (64 filters, 3×3, stride 1,1) -> Reshape to 40 patches (`5 × 8`)
   ├── `PositionalEmbedding` Layer (`max_patches=256`, dynamic sequence length addition)
   ├── ViT Encoder Blocks ($\times 3$): LayerNorm ──> MultiHeadAttention (4 heads) ──> `LayerScale` ($\gamma=10^{-2}$) ──> Add ──> LayerNorm ──> GELU FFN ──> `LayerScale` ──> Add
   ├── Global Average Pooling 1D + LayerNorm
   ├── Parallel Pearson Connectivity Path: Input (`171`) ──> Dense(32) ──> BatchNorm ──> Dropout(0.3)
   ├── Fusion & Classification Head: Concatenate [ViT Features, Connectivity Features] ──> Dense(32, GELU) ──> Dropout(0.5) ──> Softmax Output (`2` classes)

[7. Deterministic Training Sequence]
   ├── Model Compilation: AdamW optimizer (`lr=1e-4`, `weight_decay=1e-4`, `clipnorm=1.0`), Categorical Focal Loss (`gamma=2.0`), Metric: `AUC(multi_label=True)`
   ├── Callbacks: `ResourceLoggingCallback`, `CurriculumMixupCallback`, `ReduceLROnPlateau` (on `val_auc`), `ModelCheckpoint` (`weights-only`, monitor `val_auc`), `HeartbeatCheckpoint`, `CSVLogger`
   └── Execution: `model.fit()` ──> Reload best weights (`load_weights`) from disk

[8. Post-Training Evaluation & Statistical Rigor]
   ├── Fixed Thresholding (`0.5`) on Unseen Test Set
   ├── Window-Level Metrics: ROC-curve, AUROC, 95% Confidence Interval via Subject-Level Cluster Bootstrapping (1000 resamples)
   ├── Patient-Level Macro Soft-Voting: Group windows by `subject_id`, compute mean probability per patient
   ├── Patient-Level Metrics: ROC-curve, AUROC, 95% Cluster Bootstrap CI, Classification Report
   └── Artifact Persistence: Save metrics log (`final_evaluation_metrics.txt`), training dynamics, ROC curves, and confusion matrix heatmaps

```