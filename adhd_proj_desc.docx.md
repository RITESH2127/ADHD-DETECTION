## **1\. Introduction**

Attention-Deficit/Hyperactivity Disorder (ADHD) is a prevalent neurodevelopmental disorder characterized by persistent patterns of inattention, hyperactivity, and impulsivity that can substantially affect academic performance, social functioning, and daily activities. The diagnosis of ADHD is primarily based on clinical assessment, behavioral observations, developmental history, and standardized rating scales. Although these approaches are essential for clinical practice, the heterogeneous presentation of ADHD and the absence of a single definitive biological marker make objective and reliable identification challenging. Recent systematic reviews have highlighted the potential of artificial intelligence (AI), particularly machine learning (ML) and deep learning (DL), for extracting discriminative patterns from neurophysiological and neuroimaging data to support ADHD assessment. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39978669/?utm_source=chatgpt.com))

Neuroimaging modalities such as magnetic resonance imaging (MRI) and electroencephalography (EEG) provide complementary information about the neural characteristics associated with ADHD. MRI provides high-resolution structural information about the brain and can reveal subtle alterations in cortical and subcortical anatomy. Previous studies have reported ADHD-associated differences involving distributed brain regions rather than a single localized abnormality, including frontal, parietal, temporal, and subcortical structures. Machine-learning studies based on structural MRI have demonstrated the feasibility of distinguishing individuals with ADHD from healthy controls; however, performance varies considerably across datasets, preprocessing strategies, feature representations, and validation protocols. A systematic review and meta-analysis of MRI-based ML studies reported an overall pooled sensitivity of 0.74, specificity of 0.75, and an area under the curve (AUC) of 0.81, indicating promising but still insufficient diagnostic reliability for clinical deployment. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/38580035/?utm_source=chatgpt.com))

In contrast to MRI, EEG provides a non-invasive and relatively inexpensive measure of brain electrical activity with high temporal resolution. EEG-based studies have investigated differences in spectral characteristics, event-related responses, functional connectivity, and temporal dynamics between individuals with ADHD and healthy controls. Recent work has demonstrated that deep-learning models can identify discriminative EEG patterns for ADHD classification, while preprocessing and temporal segmentation can substantially influence classification performance. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/38379973/?utm_source=chatgpt.com)) However, EEG primarily captures dynamic neural activity at the scalp level and has comparatively limited spatial resolution. MRI and EEG therefore offer different but complementary perspectives: MRI characterizes **where** structural abnormalities may occur, whereas EEG provides information about **how brain activity evolves over time**.

Most existing computational approaches have focused on a single modality, which limits their ability to capture the multidimensional characteristics of ADHD. The complex and heterogeneous nature of ADHD suggests that integrating complementary neuroimaging modalities may provide a more comprehensive representation of disorder-related neural patterns. Multimodal ML has already shown potential in ADHD research, with studies combining structural, functional, and diffusion imaging demonstrating that complementary information from different imaging modalities can improve characterization of ADHD-related neural alterations. ([Nature](https://www.nature.com/articles/s41398-023-02536-w?utm_source=chatgpt.com)) Nevertheless, a recent systematic review of ML approaches using MRI and/or EEG identified substantial variation in datasets, feature extraction methods, model architectures, validation strategies, and interpretability, emphasizing the need for robust, generalizable, transparent, and reproducible multimodal diagnostic frameworks. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39978669/?utm_source=chatgpt.com))

Deep learning provides an attractive solution for multimodal ADHD detection because it can learn hierarchical representations directly from complex and high-dimensional data. Conventional convolutional neural networks (CNNs) have been extensively used for image-based neuroimaging analysis; however, their local receptive-field operations may limit the modeling of long-range dependencies within brain images. The **Vision Transformer (ViT)** addresses this limitation by dividing an image into a sequence of patches and employing self-attention to model relationships among distant regions. This capability is particularly relevant to brain MRI, where clinically meaningful patterns may involve distributed anatomical regions. Recent research has demonstrated the potential of ViT-based architectures for ADHD MRI classification, including approaches that combine transformer-based global representations with CNN-based local feature extraction. ([Nature](https://www.nature.com/articles/s41598-026-50791-w?utm_source=chatgpt.com))

The potential of ViT can be further extended to a multimodal setting by treating learned EEG and MRI representations as complementary tokens or feature embeddings and employing cross-modal attention to learn relationships between structural and electrophysiological information. Such an architecture can enable the model to learn a unified representation of ADHD by jointly considering spatial anatomical characteristics from MRI and temporal/neurophysiological characteristics from EEG. Rather than independently classifying each modality and combining the final predictions, attention-based multimodal fusion can potentially identify interactions between modalities that may not be observable from either EEG or MRI alone.

Despite these advances, several challenges remain. ADHD datasets are often limited in size, heterogeneous across acquisition sites, and affected by variations in preprocessing, demographic characteristics, and experimental protocols. Moreover, high classification performance reported under cross-validation does not necessarily guarantee generalization to independent subjects or external datasets. Previous analyses of MRI-based ADHD classifiers have specifically cautioned that conventional cross-validation can produce optimistic performance estimates compared with held-out testing, highlighting the importance of subject-independent and externally validated evaluation. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/36651494/?utm_source=chatgpt.com)) These limitations motivate the development of models that can learn robust multimodal representations while reducing dependence on handcrafted features and improving generalizability.

**In this study, we propose a multimodal Vision Transformer-based framework for ADHD detection using EEG and MRI data.** The proposed framework aims to extract modality-specific representations from EEG and MRI, employ attention-based fusion to establish cross-modal relationships, and learn a unified representation for distinguishing individuals with ADHD from healthy controls. In addition, the framework is designed to support explainable analysis by identifying the MRI regions and EEG channels that contribute to model predictions. Such an approach can potentially bridge the gap between high-performing AI models and clinically meaningful neurobiological interpretation. The proposed methodology therefore seeks to provide a robust and interpretable computational framework for multimodal ADHD detection while exploiting the complementary spatial information of MRI and temporal information of EEG.

**The major contributions of this study are threefold:** (1) development of a multimodal EEG–MRI framework based on Vision Transformer architecture for automated ADHD detection; (2) design of an attention-based multimodal fusion strategy to learn complementary relationships between electrophysiological and structural brain representations; and (3) incorporation of explainability mechanisms to identify the MRI regions and EEG channels contributing to ADHD predictions. Through the integration of complementary neuroimaging information, the proposed framework aims to improve the reliability, generalizability, and interpretability of AI-assisted ADHD detection.

### **Key references to consider**

1. Cao, M., Martin, E., & Li, X. (2023). *Machine learning in attention-deficit/hyperactivity disorder: new approaches toward understanding the neural mechanisms*. **Translational Psychiatry, 13**, 236\. ([Nature](https://www.nature.com/articles/s41398-023-02536-w?utm_source=chatgpt.com))

2. *A systematic literature review of machine learning techniques for the detection of attention-deficit/hyperactivity disorder using MRI and/or EEG data*. **Neuroscience** (2025). ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39978669/?utm_source=chatgpt.com))

3. Sanchis, J., et al. (2024). *A novel approach to identify the brain regions that best classify ADHD by means of EEG and deep learning*. **Heliyon, 10**, e26028. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/38379973/?utm_source=chatgpt.com))

4. *Structural or/and functional MRI-based machine learning techniques for attention-deficit/hyperactivity disorder diagnosis: A systematic review and meta-analysis*. (2024). ([PubMed](https://pubmed.ncbi.nlm.nih.gov/38580035/?utm_source=chatgpt.com))

5. *Machine Learning and MRI-based Diagnostic Models for ADHD: Are We There Yet?* ([PubMed](https://pubmed.ncbi.nlm.nih.gov/36651494/?utm_source=chatgpt.com))

## **Details:**

This is for **EEG \+ MRI \+ ViT \+ Grad-CAM/LIME paper**.

For **multimodal ADHD detection using EEG \+ MRI**, I would recommend:

**EEG .mat**  
→ Band-pass filtering \+ artifact removal  
→ Epoching  
→ Time–frequency transformation (STFT/Wavelet)  
→ 2D EEG representation  
→ Patch embedding  
→ **EEG-ViT**  
→ EEG feature embedding

**MRI**  
→ Preprocessing  
→ MRI-ViT  
→ MRI feature embedding

**EEG embedding \+ MRI embedding**  
→ Cross-modal attention/fusion  
→ Classifier  
→ **ADHD / Healthy Control**

### **One important point**

I would **not recommend simply reshaping the raw EEG matrix into an image** and feeding it to a conventional ViT. For a research paper, it is stronger to construct a meaningful **time-frequency or channel-time representation** so that the ViT patches have physiological significance.

If you upload **one or a few .mat files from the Frontiers ADHD dataset**, I can examine the actual structure and give you the **exact ViT preprocessing and input dimensions** for this dataset.

