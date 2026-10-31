### Preprocessing, Artifact Removal & Signal Integrity

* **Voltage Rescaling:** Multiplied raw `.mat` matrix signals by $10^{-7}$ to scale microvolt recordings to standard SI volts for numerical stability in downstream filters.


* **Channel Standardization & Montage:** Remapped non-standard legacy channel nomenclatures (`T3` $\rightarrow$ `T7`, `T4` $\rightarrow$ `T8`, `T5` $\rightarrow$ `P7`, `T6` $\rightarrow$ `P8`) and assigned the standard 10–20 spherical montage (`standard_1020`).


* **Flatline Channel Detection & Interpolation:** Added variance screening (`np.var < 1e-10`) to automatically flag disconnected or flatlined electrodes and reconstruct them using spherical spline interpolation (`raw.interpolate_bads()`).


* **Frequency Bandpass Filtering:** Configured zero-phase FIR bandpass filtering constrained strictly between $1.0\text{ Hz}$ and $40.0\text{ Hz}$ using a Hamming-windowed design (`firwin`) to eliminate DC drift and high-frequency powerline noise.


* **Common Average Referencing (CAR):** Enforced full-scalp average rereferencing (`raw.set_eeg_reference('average')`) prior to decomposition, satisfying the prerequisite mathematical assumption of the ICLabel neural network.


* **Conservative ICA Component Exclusion:** Fit extended Infomax ICA with $15$ components and integrated `mne_icalabel`, enforcing an objective $70\%$ probability cutoff (`prob > 0.70`) strictly limited to eye blinks, cardiac pulses, and muscle artifacts to prevent neural signal loss.


* **Artifact-Free Z-Score Normalization:** Replaced global channel Z-scoring with a pre-scan that calculates patient-specific mean and standard deviation solely from clean, sub-threshold windows ($\vert{}V\vert{} \le 150\ \mu\text{V}$), accompanied by a global telemetry fallback failsafe.


* **Atomic Preprocessing Caching:** Implemented disk persistence using `pickle` with `HIGHEST_PROTOCOL`, writing first to an intermediate `.tmp` file and atomically renaming it to prevent file corruption during sudden job termination.


* **Hyperparameter & Seed Fingerprinting:** Indexed cache filenames via an MD5 hash across all preprocessing hyperparameters, including `SFREQ`, `scaling_factor`, filter frequencies, artifact thresholds, and the fixed ICA random seed (`42`), preventing stale-cache contamination.


* **Dataset Integrity & ID Collision Assertion:** Enforced strict equality between discovered `.mat` files and unique group identities (`assert total_unique_groups == total_files_discovered`) to ensure identical internal `.mat` variable keys across folders do not collapse distinct subjects into a single entity.



---

### Windowing, Amplitude Thresholding & Split Management

* **Absolute Amplitude Rejection:** Replaced moving peak-to-peak amplitude difference logic with a strict absolute voltage threshold (`REJECTION_THRESHOLD = 150e-6`), instantly discarding any 2-second window containing an absolute sample deflection $> \pm 150\ \mu\text{V}$.


* **Asymmetric Segment Overlap:** Configured train windowing with an 85% step overlap (`TRAIN_OVERLAP = 0.85`) for data augmentation while enforcing 0% overlap (`TEST_OVERLAP = 0.0`) on validation and test sets to eliminate synthetic temporal correlations.


* **Leak-Free Cross-Validation:** Partitioned the dataset using `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)`, allocating 5 folds (~71%) to training, 1 fold (~14.5%) to validation, and 1 fold (~14.5%) to testing.


* **Subject-Disjoint Isolation Guardrail:** Programmed programmatic set-intersection assertions (`train_subjs.intersection(val_subjs) == ∅`, etc.) to guarantee zero patient overlap between folds.


* **Vanishing Subject Detection:** Embedded post-segmentation assertions checking that every subject assigned to a split retains at least one valid sub-$150\ \mu\text{V}$ window, preventing silent subject dropouts from distorting patient-level evaluations.


* **Training Subject ID Scope Binding:** Updated the training window segmentation call from unpacking to an ignored variable (`_`) to explicitly capturing `train_subject_ids`, eliminating a runtime `NameError` during split validation.


* **Dynamic Focal Loss Alpha Weighting:** Implemented dynamic, sample-weighted alpha vector computation ($[\alpha_{\text{Control}}, \alpha_{\text{ADHD}}]$) normalized to sum to $1.0$, counteracting minor class imbalances across extracted windows.


* **Deterministic Global Training Shuffle:** Applied a seeded global NumPy permutation (`np.random.permutation`) across concatenated window tensors prior to feeding data to the pipeline.



---

### Neural Architecture & Signal Representation

* **On-GPU STFT Spectral Representation:** Integrated a custom `GPU_STFT_Layer` computing short-time Fourier transforms directly inside the computational graph with a 128-sample window ($1.0\text{ Hz}$ resolution), 8-sample step ($93.75\%$ overlap), and 128 FFT length.


* **Stopband Truncation:** Sliced Fourier frequency bins strictly between index $0$ and $40$ ($\le 40.0\text{ Hz}$, $41$ bins total), removing all dead, non-informative stopband parameters above the low-pass filter edge.


* **Log-Decibel Energy Compression:** Mapped linear STFT magnitudes into compressed logarithmic decibels using $20.0 \times \frac{\log_{10}(\text{magnitude} + 10^{-8})}{\log_{10}(10.0)}$.


* **Factorized Spatial-Temporal Convolutions:** Designed CNN Block 1 with factorized kernels, placing a $1 \times 7$ temporal filter followed by a $7 \times 1$ spatial filter with intermediate Batch Normalization to separate channel topology from time-series features.


* **Frequency-Selective Max Pooling:** Constrained pooling layers in CNN Blocks 2 and 3 to $(2, 1)$ kernel strides, compressing frequency bins ($41 \rightarrow 20 \rightarrow 10 \rightarrow 5$) while preserving temporal frame resolution at $8$ steps to prevent premature temporal collapse.


* **Spatial Dropout Regularization:** Applied increasing 2D Spatial Dropout across convolution blocks ($0.3 \rightarrow 0.3 \rightarrow 0.4$) to drop entire feature map channels rather than isolated pixels.


* **Patch Grid Tokenization:** Project feature maps through a Conv2D layer ($64$ filters, $3 \times 3$, stride 1) and reshaped the resulting $5 \times 8$ grid into $40$ distinct sequential tokens of dimension $D_{\text{model}} = 64$.


* **Dynamic Fixed-Capacity Positional Embedding:** Replaced broken static/functional embeddings with an explicit `PositionalEmbedding` layer initialized with `max_patches=256`, dynamically indexing positions via `tf.shape(inputs)[1]` to prevent shape mismatches across variable input lengths.


* **LayerScale Parameter Initialization:** Replaced standard residual paths with a custom `LayerScale` layer initialized to $\gamma = 10^{-2}$, stabilizing deep gradient backpropagation across the 3 Vision Transformer blocks.


* **Pre-Norm Multi-Head Attention:** Built 3 Transformer encoder blocks utilizing Pre-Layer Normalization ($\epsilon = 10^{-6}$), 4 attention heads, key dimension of $32$, and an attention dropout rate of $0.3$.


* **GELU Position-Wise Feed-Forward Network:** Expanded intermediate token representations by $2\times$ ($64 \rightarrow 128$) using Gaussian Error Linear Units (`gelu`) before projecting back to $D_{\text{model}} = 64$.


* **Keras Layer Serialization:** Registered all custom layers (`GPU_STFT_Layer`, `LayerScale`, `PositionalEmbedding`) with `@tf.keras.utils.register_keras_serializable()` and defined explicit `get_config()` dictionaries for serializing and reloading models.


* **Parallel Functional Connectivity Branch:** Calculated upper-triangular Pearson correlation vectors ($171$ features) outside the STFT branch, embedding them through `Dense(32, ReLU) -> BatchNorm -> Dropout(0.3)`.


* **Multimodal Late Feature Concatenation:** Combined global average pooled ViT features ($64$-d) and connectivity embeddings ($32$-d) into a unified $96$-dimensional vector feeding into `Dense(32, GELU) -> Dropout(0.5) -> Dense(2, Softmax)`.


* **Solves Phase Destruction:** Computes the $171$-feature Pearson connectivity vector on each clean, unmixed window during segmentation in Cell 4. Mixup now interpolates connectivity matrices directly ($Conn_{\text{mixed}} = \lambda Conn_A + (1-\lambda)Conn_B$) instead of calculating correlations on phase-interfered raw signals.   

* **Guarantees 100% Determinism:** Eliminates tf.numpy_function, NumPy PRNG calls, and multithreading race conditions across background workers in Cell 5. Mixup and Beta sampling ($\frac{\Gamma_1}{\Gamma_1 + \Gamma_2}$) run in native TensorFlow operations bound to set_random_seed(42) and enable_op_determinism().   

* **Optimizes Throughput:** Bypasses dynamic runtime Pearson calculation on CPU workers, feeding precomputed tensors directly into GPU memory.   

* **Resolution of Overfit**: sensitivity tuned for train dynamics: lower overlap, higher weight decay, lower patience lr, higher vit dropout. also tested mixup, but that resulted in underfit.

---

### Optimization, Checkpointing & Diagnostics

* **Gradient-Clipped AdamW:** Compiled the model using AdamW with decoupled weight decay (`1e-4`), initial learning rate (`1e-4`), and global $L_2$ gradient norm clipping (`clipnorm=1.0`) to avoid exploding gradients.


* **Multi-Label Metric Configuration:** Configured tracking using `tf.keras.metrics.AUC(name='auc', multi_label=True)`, avoiding scalar metric dimension mismatches under one-hot targets.


* **Plateau-AUROC Alignment:** Aligned `ReduceLROnPlateau` to monitor `val_auc` (`mode='max'`, `factor=0.5`, `patience=5`) instead of `val_loss`, preventing premature learning rate cuts when focal loss oscillations occur alongside improving AUROC.


* **Weights-Only Model Checkpointing:** Updated `ModelCheckpoint` to monitor `val_auc` (`mode='max'`, `save_best_only=True`) and configured `save_weights_only=True` saving to `.weights.h5`, bypassing custom loss closure deserialization errors.


* **Best Validation Weight Restoration:** Programmed an explicit reload step (`model.load_weights(best_weights_path)`) immediately after training completes to ensure downstream test evaluation runs on optimal model weights.


* **Throttled Heartbeat Checkpointing:** Restricted periodic full-graph checkpointing (`HeartbeatCheckpoint`) from saving every 5 epochs to every 15 epochs, minimizing disk I/O overhead over 50–60 epoch runs.


* **Hardware Resource Logging:** Added `ResourceLoggingCallback` to profile execution time per epoch alongside tracking peak GPU VRAM allocations via `tf.config.experimental.get_memory_info('GPU:0')`.


* **Curriculum Mixup Controller:** Built an on-epoch-begin mixup callback delaying sample interpolation until `MIXUP_START_EPOCH = 30` with mild blending ($\alpha = 0.2$), keeping early boundary learning focused on unmixed representations.



---

### Statistical Rigor & Zero-Leakage Evaluation

* **Pre-Set Non-Tuned Decision Threshold:** Replaced post-hoc Youden's J test-set threshold searching with an invariant, pre-registered cutoff of $0.5$ across all test sets, eliminating threshold-tuning data leakage.


* **Patient-Level Macro Soft-Voting:** Grouped all test window prediction vectors by unique `subject_id` and took their arithmetic mean, evaluating clinical diagnostic performance on a patient basis rather than on correlated windows.


* **Subject-Level Cluster Bootstrapping:** Replaced standard independent sample bootstrapping with a cluster bootstrap function (`n_bootstraps=1000`, seeded RNG) that resamples whole patient identities with replacement, preserving intra-subject window correlation structures.


* **Empirical 95% Confidence Intervals:** Extracted percentile bounds directly from sorted bootstrap distributions ($2.5\text{th}$ and $97.5\text{th}$ percentiles) for both window-level and patient-level AUROC curves.


* **Zero-Overlap Overview Plots:** Restamped clinical visualization logic in Section 4 to trigger only on 0% overlap sets, preventing artificial time-axis distortion in diagnostic validation plots.


* **Evaluation Telemetry Output:** Configured automated plain-text logging (`final_evaluation_metrics.txt`), writing window and patient classification reports, exact AUROCs, confidence bounds, confusion matrices, and ROC curves to disk.