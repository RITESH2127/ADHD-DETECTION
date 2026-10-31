## Academic Refinements Already Done:

1. determinism and shuffle

2. fixed a lot of things in pre processing
- fixed ica
- added bad electrode interpolation
- added the absolute window threshold, also tested ptp
- some z score thing?
- scaling fix with exp
- stft frame and fft length increased
- order : filter before CAR
- stratified

3. Master Mixup Controls: disabled mixup
The code introduces definitive toggle controls in the Config class. You can set ENABLE_MIXUP = False to run a pure, clean baseline. If left active alongside CURRICULUM_MIXUP = True, the custom CurriculumMixupCallback keeps the Mixup alpha strictly disabled (0.0) for the first 30 epochs, allowing the model to rapidly learn sharp clinical boundaries before gently blending the representations to smooth the ROC curve.

4. Implicit Regularization via GPU-STFT
The model fully deprecates the CPU-bound Morlet CWT mapping in favor of the GPU_STFT_Layer. By moving the spectrogram extraction directly into the Keras Graph, processing latency drops dramatically, and the lower-resolution discretization natively protects the Vison Transformer from overfitting to granular patient noise artifacts.

5. Bootstrapping Confidence Intervals
Single-point estimates are scientifically unreliable. The implementation now executes compute_bootstrap_auc against 1,000 independent samples of the testing distribution to calculate the 95% Confidence Interval.

6. Patient-Level Clinical Soft-Voting
Because a 2-second overlapping window is a dataset construction rather than a clinical entity, the segment_data_with_subjects loop explicitly traces the subject_id for every tensor batch. During evaluation, the softmax probabilities are aggregated through a pandas dataframe, applying a macro-voting filter to neutralize transient biological noise before performing a secondary patient-level Youden's J threshold calibration.

7. removed early stopping but added use best model for proper benchmarking

fix 4 and 5 FROM claude thing


### Direct Defend:

**1. DEFEND, just a thing to be mindful of: Retrospective Temporal Leakage (Z-Scoring)**

* **The Flaw:** Calculating `subject_mean` and `subject_std` across the *entire* continuous EEG recording means the normalization of Window 1 includes information from Window 1000. While sometimes accepted in retrospective studies, clinical deployment reviewers (e.g., *Nature Medicine*) often flag this because a real-time hospital EEG cannot calculate a mean using future data.
* **The Fix:** Acknowledge this as a limitation in the paper ("offline retrospective normalization"), or switch to a running standardization window (e.g., Z-scoring each 2-second window independently, or using a 10-second trailing historical mean).

**2. DEFEND: Broadband Pearson Connectivity (The "Averaged" Topography)**

In the parallel `tf.data` pipeline, you calculate the Pearson phase-locking matrix using `np.corrcoef` on the raw 1-40 Hz signal.

* **The Flaw:** You are asking the model: "How correlated is the front of the brain (Fp1) with the back of the brain (O1) overall?"
* **The Biological Problem:** In ADHD, brain connectivity is highly frequency-dependent. A patient might have severe *hyper-connectivity* in the Theta band, but *hypo-connectivity* in the Beta band. Because you are calculating Pearson on the entire 1-40 Hz broadband signal at once, these two opposing effects average each other out. The resulting matrix hides the very disease markers you are trying to find.
* **The Fix:** The true "State-of-the-Art" approach is to calculate the Pearson matrix on specific frequency bands (e.g., apply a Theta filter, calculate Pearson; apply a Beta filter, calculate Pearson) and feed a multi-channel matrix into the network. If you leave it as broadband, you must explicitly acknowledge this in your paper as a "broadband topological limitation."

**3. (Recording length as a confound):** 

In your patient-level macro-voting evaluation (Cell 8), df_results.groupby('subject_id').mean() gives every test patient 1 equal vote, regardless of whether their recording is 100 seconds or 300 seconds. Thus, recording length cannot bias your patient-level test metrics. Just log the mean recording duration for ADHD vs. Control in Cell 3 to report in a table.

**4. (Within-subject Z-scoring on test patients):**

Computing a test subject's Z-score mean and standard deviation from their own clean recording is not data leakage. No class labels are used, and zero information crosses between subjects. In a real hospital deployment, a new patient's 3-minute EEG recording is normalized against their own baseline voltage before window classification. State this explicitly as "subject-specific calibration to eliminate inter-subject skull impedance variance."



