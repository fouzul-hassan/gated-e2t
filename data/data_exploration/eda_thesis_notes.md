# EDA Notes for Thesis

This document gives short thesis-ready descriptions and insights for the two EDA notebooks:
- `eda_zuco_dataset.ipynb`
- `eda_brain_topomaps.ipynb`

The wording below is intentionally concise so it can be adapted directly into the thesis EDA chapter.

---

## 1. `eda_zuco_dataset.ipynb`

| Section | Short Description | Thesis Insight |
|---|---|---|
| 1. Load Data | Loads the intermediate EEG data and the final preprocessed dataset. | This establishes the data pipeline from raw/intermediate EEG tensors to the final learning-ready dataframe. It confirms that the final dataset preserves the expected sample structure while adding label and split information. |
| 2. Dataset Overview & Schema | Prints the full schema and examples of each column type. | The schema inspection confirms that the dataset contains EEG arrays, masks, task labels, subject IDs, text fields, and derived metrics, making it suitable for both modelling and analysis. |
| 3. Dataset & Task Distribution | Shows how samples are distributed across datasets and reading tasks. | The dataset is not evenly distributed across tasks, so task-aware analysis is necessary. This also motivates reporting results separately by task rather than relying on aggregate statistics alone. |
| 4. Train / Validation / Test Split | Visualises the split and how it is distributed across tasks. | A balanced split is important for fair model evaluation. The figure helps verify that each task appears in all partitions and that the final split supports supervised training and testing without obvious leakage. |
| 5. Subject-Level Analysis | Shows sample counts per subject and per subject-task combination. | Subject-wise variation is substantial, so the model must generalise across individuals rather than memorising subject-specific patterns. This also suggests that subject identity may be a strong latent factor in the EEG representation. |
| 6. EEG Signal Analysis (Before vs After Preprocessing) | Displays example EEG traces after preprocessing. | The traces illustrate that the EEG signals remain time-varying and channel-specific after cleaning and alignment. This supports the use of a sequence model rather than a simple summary-statistics approach. |
| 7. EEG Statistics & Distribution | Plots mean, standard deviation, min/max, and channel-wise statistics. | The statistics show that the preprocessed EEG signals are centred near zero but still contain meaningful amplitude variability. Outliers and wide variance are expected in EEG and should be interpreted as part of the physiological signal plus residual artefact. |
| 8. EEG Heatmap (Time × Channel) | Visualises the EEG matrix as a heatmap for selected samples. | The heatmaps provide a compact view of temporal structure and cross-channel variation. They make it clear that only part of the padded tensor contains valid information, reinforcing the need for masking. |
| 9. Mask Analysis (Zero-Padding) | Shows the distribution of actual lengths, padding ratios, and durations. | The padding analysis confirms that zero-padding is extensive and variable across trials. This is a strong justification for using a validity mask in the model so that attention and loss do not treat padded values as real signal. |
| 10. Text Analysis | Shows text length distributions and the relation between EEG duration and text length. | Text length varies across tasks but remains within a manageable range for generation. The comparison with EEG duration suggests that reading behaviour is not uniform and must be handled as a variable-length sequence-to-sequence problem. |
| 11. Label Analysis (Sentiment & Relation) | Visualises class balance for sentiment and relation labels. | The label distributions reveal that some classes are more frequent than others. This means classification-style evaluation should consider class imbalance and not rely on accuracy alone. |
| 12. Text Variants (Augmentations) | Shows multiple rewritten versions of the same sentence and their average lengths. | The augmented text variants demonstrate that each EEG sample is paired with multiple linguistic forms, which is useful for robustness testing and for examining how wording variation affects generation quality. |
| 13. Cross-Subject EEG Comparison | Compares average EEG mean and standard deviation across subjects. | This section highlights strong subject-to-subject variability in the EEG signal. The result supports the need for a model that can learn shared structure while still being robust to individual neural differences. |
| 14. EEG Channel Correlation Matrix | Plots average correlation between EEG channels. | The channel correlation matrix indicates that the channels are not independent and that spatial structure exists across the scalp. This justifies using multi-channel models that can exploit inter-channel dependencies. |
| 15. EEG Power Spectral Density (PSD) | Shows average spectral content and task-wise PSD curves. | The PSD plot confirms that useful information is distributed across frequency bands rather than concentrated in a single narrow range. This supports frequency-aware EEG modelling and band-sensitive interpretation. |
| 16. Summary Statistics Table | Prints a compact dataset summary. | The summary provides a thesis-friendly overview of dataset scale, splits, channel count, tasks, and label variety. It is useful as a final dataset snapshot before model training. |
| 17. Before vs After Preprocessing Comparison | Compares raw/intermediate and final preprocessed data. | This comparison shows exactly what preprocessing contributed: downsampling, padding, masking, and label matching. It demonstrates that the final dataset is aligned with the prototype requirements while preserving the underlying EEG structure. |

### Short thesis paragraph for this notebook
The EDA of the ZuCo dataset shows that the final preprocessed EEG-text corpus is structured, multi-task, and variable in both sequence length and subject-specific signal characteristics. The zero-padding analysis confirms that masking is essential, while the spectral, channel-correlation, and subject-level plots show that EEG contains meaningful temporal, spatial, and frequency-domain structure. Together, these observations justify the use of a masked sequence model with multi-channel attention and task-aware evaluation.

---

## 2. `eda_brain_topomaps.ipynb`

| Section | Short Description | Thesis Insight |
|---|---|---|
| Setup: EGI Channel Montage | Defines the EEG sensor montage and channel positions for scalp plotting. | This ensures that the topographic maps are aligned with a realistic EEG sensor layout, making the plots interpretable in terms of scalp location. |
| Helper Functions | Computes band power and prepares reusable topomap plotting utilities. | The helper functions convert raw EEG recordings into scalp-level summaries, enabling consistent comparison across tasks, subjects, and conditions. |
| 1. Frequency Band Power — Overall Brain Topomaps | Shows average scalp power for delta, theta, alpha, beta, and gamma bands. | The overall topomaps reveal that EEG activity is not spatially uniform. Different frequency bands emphasise different scalp regions, so frequency-specific analysis is necessary rather than relying on a single aggregate map. |
| 2. Band Power Comparison Across Tasks | Compares band power across Sentiment Reading, Normal Reading, and Relation Reading. | This section shows that reading task changes the spatial distribution of EEG power. It suggests that cognitive state and task demands modulate neural activity in a band-specific way. |
| 3. Task Difference Maps | Subtracts Normal Reading from the other tasks to highlight task-specific changes. | The difference maps isolate task-dependent neural effects more clearly than the raw maps. They are useful for thesis discussion because they show where Sentiment Reading and Relation Reading diverge from the baseline condition. |
| 4. Sentiment Condition Brain Maps | Shows band power maps for different sentiment labels within Task 1. | The sentiment maps suggest that emotional or evaluative reading conditions may produce distinct scalp patterns, though the effect is likely subtler than the task-level differences. |
| 5. Sentiment Difference Maps | Compares sentiment labels against a baseline sentiment condition. | This figure is useful for showing within-task variation, but the differences are expected to be smaller than task-level contrasts. It demonstrates that the EEG representation can still capture finer label-dependent structure. |
| 6. Relation Type Brain Maps | Shows brain maps for the most frequent relation labels in Task 3. | The relation plots indicate that the EEG signal contains information related to semantic relation processing. This supports the idea that the model is not just learning generic reading activity, but task-specific comprehension patterns. |
| 7. Cross-Subject Brain Maps | Compares alpha-band power for selected subjects. | The plots highlight that different participants can have noticeably different neural baselines and spatial patterns. This reinforces the need for subject-robust modelling and careful cross-subject evaluation. |
| 8. Single Trial Brain Maps | Visualises individual EEG recordings instead of averages. | Single-trial topomaps show that the dataset retains trial-level variability, which is important because the model must operate on noisy individual recordings rather than only on averaged signals. |
| 9. Average EEG Amplitude Topomaps | Maps mean absolute amplitude and standard deviation across the scalp by task. | The amplitude maps show where signal strength and variability are concentrated across the scalp. They provide a simple but useful bridge between raw EEG statistics and spatial interpretation. |
| 10. ZuCo1 vs ZuCo2 Brain Comparison | Compares scalp power between the two recording datasets. | This comparison suggests that acquisition session or dataset origin can change the EEG baseline. That means dataset-specific effects should be considered when interpreting results across subjects or sessions. |

### Short thesis paragraph for this notebook
The brain topomap analysis shows that EEG activity has clear spatial and frequency-specific structure across the scalp. Task-based comparisons indicate that reading conditions modify the distribution of power across bands, while subject and dataset comparisons reveal substantial baseline variation. These results support the use of topographic EEG visualisation in the thesis and justify modelling choices that preserve multi-channel spatial information.

---

## Optional thesis phrasing for the EDA chapter
The exploratory analysis confirmed that the ZuCo EEG-text data is variable across subjects, tasks, and frequency bands, with meaningful spatial patterns visible on the scalp. The dataset also contains substantial zero-padding, making mask-aware modelling necessary. Across both notebooks, the visualisations collectively show that the EEG signal retains temporal, spectral, and spatial structure that can support downstream EEG-to-text modelling.
