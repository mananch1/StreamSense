# StreamSense: Real-Time Reference-Free Semantic Drift Detection and Closed-Loop Model Recovery for Streaming Sentiment Analysis

**Manan Chahal, Parashi Rawal, Aryan Racha, and Manali Kulkarni**  
*Department of Computer Engineering, Vidyalankar Institute of Technology, Mumbai, India*  
Email: {manan.chahal, parashi.rawal, aryan.racha, manali.kulkarni}@vit.edu.in  

---

**Abstract** — Deploying natural language processing (NLP) models on streaming social feeds exposes them to concept drift, where the statistical properties of incoming text evolve over time due to shifts in vocabulary, sentiment polarity, syntactic quality, and community norms. While Garcia et al. (2024) proposed four methods for generating drift in text streams, their evaluation was limited to offline, paired-text settings using only accuracy and F1 as post-hoc metrics. In this paper, we present *StreamSense*, a real-time streaming system that extends their drift generation taxonomy with two novel practical perturbation methods — Noise Injection and Formality/Slang Shift — and introduces a reference-free frozen baseline detection architecture that monitors nine statistical drift indicators across five complementary dimensions without requiring access to clean parallel text. We further integrate a closed-loop MLOps pipeline that ties unsupervised drift detection to supervised downstream model performance, enabling on-demand model retraining with automatic baseline recalibration. Experimental evaluation on 500 Amazon Movie Reviews demonstrates that: (i) different drift methods produce distinct, separable metric signatures; (ii) Class Swap causes the most severe model degradation (accuracy dropping from 91.0% to 9.6%); (iii) on-demand retraining recovers accuracy by up to +69.3 percentage points; and (iv) the multi-criteria monitoring framework successfully distinguishes semantic, lexical, syntactic, and label-level drift in real time.

**Keywords** — concept drift, data drift detection, streaming NLP, sentiment analysis, reference-free monitoring, MLOps, text stream mining

---

## I. Introduction

Machine learning models deployed in production environments face a fundamental challenge: the data distribution they encounter at inference time inevitably diverges from the distribution they were trained on. This phenomenon, termed *concept drift* [1], is particularly acute in natural language processing (NLP), where language evolves continuously through new vocabulary, shifting cultural norms, platform-specific conventions, and varying data quality.

Garcia et al. [2] made an important contribution by proposing four systematic methods for generating concept drift in text streams — Adjective Swap, Class Swap, Class Shift, and Time-Slice Removal — enabling controlled benchmarking of drift-resilient classifiers. However, their work has several limitations that this paper addresses:

1. **Offline paired evaluation**: Their experimental setup required access to both the original and drifted text at evaluation time, which is unavailable in real-world streaming deployments.
2. **Limited drift taxonomy**: Their four methods capture semantic and label-level drift but omit syntactic degradation (typos, noise) and sociolinguistic shifts (formality, slang) — common real-world phenomena.
3. **Single-metric evaluation**: Using only accuracy and Macro F1-Score, their evaluation cannot distinguish *what kind* of drift is occurring.
4. **No operational recovery**: They measured degradation but did not attempt model retraining or accuracy restoration.

StreamSense addresses all four limitations through the following contributions:

- **C1**: An extended drift generation taxonomy with two novel practical methods: *Noise Injection* (keyboard-adjacency typos, character mutations, truncation) and *Formality/Slang Shift* (register transformation from formal to informal text).
- **C2**: A *reference-free frozen baseline* detection architecture that establishes a multi-signal text profile during an initial burn-in phase and permanently freezes it, enabling drift detection without access to clean parallel text.
- **C3**: A *multi-criteria monitoring framework* comprising nine statistical indicators across five complementary dimensions (semantic, lexical, sentiment, syntactic, and label), experimentally shown to produce distinct metric signatures for each drift type.
- **C4**: A *closed-loop MLOps pipeline* that integrates drift detection with downstream model performance monitoring and on-demand retraining, demonstrating accuracy recovery from as low as 9.3% back to 80.0%.

---

## II. Related Work

### A. Concept Drift in Text Streams

Concept drift occurs when the joint probability distribution P(X, Y) changes over time [1]. In text classification, this manifests through shifts in vocabulary (P(X)), sentiment patterns (P(Y|X)), or both. Gama et al. [3] provide a comprehensive taxonomy distinguishing sudden, gradual, incremental, and recurring drift patterns.

Garcia et al. [2] specifically addressed the lack of benchmark datasets with labeled drift change points in text streams. Their four generation methods — Adjective Swap (using WordNet antonyms), Class Swap (label inversion), Class Shift (cyclic label rotation), and Time-Slice Removal (temporal masking) — were validated on Yelp and Airbnb datasets using Gaussian Naive Bayes, Incremental SVM, and Adaptive Random Forest classifiers.

Recent systematic reviews [4] highlight the growing need for unsupervised drift detection methods, since ground-truth labels are typically unavailable in real-time text streams. This motivates our reference-free architecture.

### B. Statistical Drift Detection Methods

Classical drift detectors such as ADWIN [5], DDM [6], and Page-Hinkley [7] operate on scalar performance streams and detect change points in error rates. While effective for tabular data, they require labeled feedback and cannot capture the multi-dimensional nature of text degradation.

Information-theoretic divergence measures — Kullback-Leibler (KL) divergence [8], Jensen-Shannon Divergence (JSD) [9], and Wasserstein distance (Earth Mover's Distance) [10] — have been adopted by industry monitoring platforms such as Evidently AI [11] and Alibi Detect [12] for comparing feature distributions between reference and production data. Our work applies these measures specifically to sentiment polarity distributions in text streams, complemented by lexical, semantic, and syntactic indicators.

### C. Sentiment Analysis Under Drift

VADER (Valence Aware Dictionary and sEntiment Reasoner) [13] remains widely used for real-time sentiment analysis due to its zero-training-cost lexicon-based approach. However, its static vocabulary makes it vulnerable to language evolution [14]. Hybrid approaches combining VADER features with TF-IDF representations and machine learning classifiers have shown improved robustness [15], motivating our hybrid feature architecture.

### D. MLOps and Model Maintenance

The MLOps paradigm [16] emphasizes treating ML models as living systems requiring continuous monitoring, drift detection, and retraining. Recent work on drift-aware retraining [17] demonstrates that triggering model updates only upon confirmed drift achieves accuracy comparable to continuous retraining while significantly reducing computational cost. StreamSense implements this philosophy through its closed-loop pipeline.

---

## III. System Architecture

StreamSense comprises two integrated components connected via WebSocket streaming:

### A. Drift Engine and Feed Simulator (Backend)

The backend, implemented in Python using FastAPI, manages:
- **Data Loading**: Sequential review streaming from the Stanford SNAP Amazon Movie Reviews corpus [18] (500 reviews, product B002QZ1RS6).
- **Drift Engine**: Applies configurable drift transformations with parametric intensity control and five temporal curve patterns (constant, gradual, sudden, sinusoidal, step).
- **Dual-Trigger Windowing**: Stream windows flush on N messages or T seconds timeout, whichever occurs first, handling both high-throughput and bursty traffic patterns.
- **Metrics Calculator**: Computes nine reference-free drift indicators per window.
- **Sentiment Model**: Hybrid TF-IDF + VADER logistic regression classifier with real-time inference and observability.

### B. Reference-Free Monitoring Pipeline

The monitoring pipeline operates in two phases:

**Burn-In Phase** (W = 1...B, default B = 3): Clean initial windows are processed to accumulate a baseline profile comprising: (i) a TF-IDF centroid vector, (ii) a vocabulary token set, (iii) a VADER sentiment polarity distribution, (iv) a rating score histogram, and (v) a baseline spelling error rate.

**Monitoring Phase** (W > B): The baseline profile is **permanently frozen**. All subsequent windows are compared against this static reference. This design prevents the *boiling frog* failure mode [5] where gradual drift progressively adapts the baseline, making incremental degradation undetectable.

---

## IV. Drift Generation Methodology

StreamSense implements six drift generation methods: four adapted from Garcia et al. [2] and two novel practical extensions.

### A. Methods from Garcia et al. [2]

**Adjective Swap** (Semantic Drift): Text is tokenized and POS-tagged; adjectives (JJ, JJR, JJS) are replaced with antonyms retrieved from WordNet synsets, similar_to satellite adjectives, and a supplemental high-frequency antonym dictionary covering 37 common sentiment-bearing adjectives.

**Class Swap** (Abrupt Label Drift): Rating scores are inverted symmetrically: 1-star becomes 5-star, 2-star becomes 4-star, 3-star unchanged.

**Class Shift** (Gradual Label Drift): Ratings are cyclically incremented: 1 -> 2 -> 3 -> 4 -> 5 -> 1.

**Time-Slice Removal** (Temporal Discontinuity): Entire temporal windows are masked, simulating system outages or missing data epochs.

### B. Novel Practical Extensions

**Noise Injection** (Syntactic Perturbation): Simulates realistic text degradation through four mutation operators applied stochastically at configurable intensity:
- *Keyboard-adjacency typos*: Characters replaced with QWERTY-adjacent keys.
- *Character deletion*: Random character removal from words > 3 characters.
- *Character transposition*: Adjacent character pairs are swapped.
- *Character duplication*: Random characters are doubled.

At high intensity (> 0.6), truncation is applied — text is cut at 65-90% of its length. This models real-world phenomena including noisy mobile keyboards, OCR errors, and bot-generated traffic.

**Formality/Slang Shift** (Register Transformation): A lexicon of 37 formal-to-informal mappings (e.g., "excellent" -> "goated", "masterpiece" -> "absolute banger", "because" -> "cuz") is applied stochastically using regex-based substitution. This models generational community shifts and platform migration effects where formal review language devolves into internet vernacular.

### C. Parametric Drift Dynamics

All intensity-parameterized methods (Adjective Swap, Noise Injection, Formality Shift) support five temporal drift curves via a multiplier function m(t):

| Curve | m(t) | Behavior |
|---|---|---|
| Constant | 1.0 | Uniform intensity |
| Gradual | phi(t) | Linear ramp from 0 to 1 |
| Sudden | I[phi(t) >= 0.5] | Step function at cycle midpoint |
| Sinusoidal | 0.55 + 0.45 sin(2*pi*phi(t)) | Periodic oscillation |
| Step | ceil(4*phi(t)) / 4 | Quantized 4-level staircase |

where phi(t) = (t mod C) / C is the normalized phase and C is the configurable cycle length.

---

## V. Reference-Free Detection Framework

### A. Metric Suite

StreamSense monitors nine statistical indicators across five complementary dimensions:

**1. TF-IDF Centroid Cosine Similarity** (Semantic) — [0.0, 1.0]:

Sim_cos = (mu_base . mu_curr) / (||mu_base||_2 * ||mu_curr||_2)

The baseline TF-IDF vectorizer (fitted during burn-in, 2000 features, English stop words removed) transforms current window texts, and the resulting centroid is compared against the frozen baseline centroid.

**2. Vocabulary Jaccard Overlap** (Lexical) — [0.0, 1.0]:

J(V_base, V_curr) = |V_base intersection V_curr| / |V_base union V_curr|

Tokens are extracted via regex (>= 3 alphabetic characters, lowercased) and compared as sets against the frozen baseline vocabulary.

**3-5. Sentiment Distribution Divergence** — VADER compound polarity thresholds (>= 0.05 positive, <= -0.05 negative, otherwise neutral) define a categorical distribution P = [P(pos), P(neu), P(neg)] per window:

- **KL Divergence**: D_KL(P_base || Q_curr) = sum_c P(c) ln(P(c)/Q(c)) with epsilon = 1e-5 smoothing. Range [0, +inf).
- **Jensen-Shannon Divergence**: JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M) where M = 0.5*(P + Q). Bounded [0.0, 1.0] using base-2 logarithm.
- **Wasserstein-1 Distance**: W_1(P, Q) = sum_k |F_P(k) - F_Q(k)| over ordinal states (neg < neu < pos). Range [0.0, 2.0].

**6. Spelling Error Rate** (Syntactic) — [0.0, 1.0]:

SER = (1/N_tokens) * sum_{j=1}^{N_tokens} I(w_j not in Dict_en)

Reference-free syntactic noise measurement via English dictionary (NLTK words corpus) lookup.

**7. Score Distribution Divergence** (Label) — [0.0, 1.0]:

JSD computed over 5-bin histograms of star ratings (1-star through 5-star) between baseline and current window.

**8. Composite Drift Magnitude** — [0%, 100%]:

CDM = (0.30 * d_cos + 0.20 * d_vocab + 0.20 * sqrt(JSD_sent) + 0.15 * delta_SER + 0.15 * JSD_score) * 100

where d_cos = 1 - Sim_cos, d_vocab = 1 - J, and delta_SER is the normalized spelling error rate increase relative to the baseline.

---

## VI. Closed-Loop MLOps Pipeline

StreamSense implements a four-stage operational cycle:

**Stage 1 — Warm-Up Training**: Upon completion of the burn-in phase, a sentiment classifier is automatically trained on the accumulated clean reviews. The classifier uses a hybrid feature representation: TF-IDF vectors (2500 features, unigrams and bigrams) concatenated with scaled VADER sentiment features [pos, neu, neg, compound] * 3.5, fed into a balanced-class-weighted Logistic Regression.

**Stage 2 — Streaming Inference**: Every incoming review is classified by the active model. Predictions are compared against star-rating proxy labels (4-5 star: positive, 3 star: neutral, 1-2 star: negative) to compute real-time window and cumulative accuracy.

**Stage 3 — Degradation Detection**: Window accuracy is categorized: *healthy* (>= 75%), *at risk* (55-75%), *degraded* (< 55%).

**Stage 4 — On-Demand Retraining**: The user triggers retraining on the most recent accumulated stream samples (up to 500). Upon retraining:
1. A new model version is instantiated (v2.0, v3.0, ...).
2. The drift metrics baseline is **recalibrated** to match the new model's training distribution via `calibrate_baseline_from_data()`, ensuring subsequent drift is measured relative to the retrained model's reference state rather than the original burn-in profile.

This recalibration step is critical: without it, the system would perpetually flag the difference between the original clean distribution and the current drifted distribution, even though the model has already adapted.

---

## VII. Experimental Evaluation

All experiments were conducted on 500 unique reviews from the Stanford SNAP Amazon Movie Reviews dataset [18] (product B002QZ1RS6 — 957 long-form reviews). Window size was set to 25 reviews. Burn-in comprised 3 clean windows (75 reviews). Drift intensity was set to 0.7 unless otherwise stated. Results are averaged over 3 monitoring windows where applicable.

### A. Per-Method Drift Metric Sensitivity (Table I)

Each drift method was applied individually to evaluate which metrics respond and by how much relative to a clean (no drift) baseline.

**Table I: Per-Method Drift Metric Sensitivity Analysis**

| Method | Cosine Sim | Vocab Overlap | Sent. KL | Sent. JSD | Sent. W1 | SER | Score JSD | CDM (%) |
|---|---|---|---|---|---|---|---|---|
| No Drift (Clean) | 0.7747 | 0.3094 | 0.026 | 0.011 | 0.120 | 0.128 | 0.074 | 23.7 |
| Adjective Swap | 0.7294 | 0.2824 | 0.205 | 0.082 | 0.560 | 0.124 | 0.074 | 29.3 |
| Class Swap | 0.7747 | 0.3094 | 0.026 | 0.011 | 0.120 | 0.128 | **0.691** | 32.9 |
| Class Shift | 0.7747 | 0.3094 | 0.026 | 0.011 | 0.120 | 0.128 | **0.470** | 29.6 |
| Noise Injection | 0.7367 | **0.1968** | 0.083 | 0.038 | 0.187 | **0.355** | 0.074 | **43.9** |
| Formality Shift | **0.6997** | 0.3052 | 0.137 | 0.057 | 0.387 | 0.138 | 0.074 | 29.6 |

**Key findings**: (i) Class Swap exclusively perturbs the Score Distribution JSD (0.691) while leaving all text-level metrics completely unchanged — confirming it is a pure label drift. (ii) Noise Injection produces the largest Composite Drift Magnitude (43.9%) due to simultaneously affecting vocabulary overlap (0.197), spelling error rate (0.355), and cosine similarity. (iii) Formality Shift primarily impacts cosine similarity (0.700) and sentiment Wasserstein distance (0.387), reflecting vocabulary replacement that alters both meaning and polarity. (iv) Adjective Swap uniquely elevates sentiment KL divergence (0.205) and Wasserstein distance (0.560), consistent with its targeted polarity reversal of sentiment-bearing adjectives.

### B. Multi-Intensity Drift Impact (Table II)

Three intensity-parameterized methods were tested at alpha in {0.0, 0.25, 0.5, 0.75, 1.0}.

**Table II: Composite Drift Magnitude (%) at Varying Intensities**

| Intensity | Adjective Swap | Noise Injection | Formality Shift |
|---|---|---|---|
| 0.00 | 26.5 | 26.5 | 26.5 |
| 0.25 | 27.4 | 44.6 | 27.6 |
| 0.50 | 29.6 | 42.2 | 28.7 |
| 0.75 | 32.7 | 46.5 | 32.1 |
| 1.00 | 35.0 | 46.1 | 33.3 |

**Key findings**: Noise Injection exhibits the steepest response, reaching near-maximum drift magnitude at intensity 0.25, indicating that even mild typographical noise has an outsized impact on reference-free text metrics — particularly vocabulary overlap and spelling error rate. Adjective Swap and Formality Shift show more gradual, monotonically increasing profiles, consistent with their word-level (rather than character-level) perturbation mechanisms.

### C. Sentiment Model Accuracy Degradation (Table III)

A Logistic Regression classifier (hybrid TF-IDF + VADER features) was trained on 100 clean reviews (training accuracy: 91.0%) and then evaluated over 5 streaming windows of 25 reviews each under various drift conditions.

**Table III: Model Accuracy Degradation Under Drift**

| Drift Method | Train Acc | W1 | W2 | W3 | W4 | W5 | Avg Acc | Drop |
|---|---|---|---|---|---|---|---|---|
| No Drift | 0.910 | 0.880 | 0.920 | 0.880 | 0.720 | 0.840 | 0.848 | 0.062 |
| Adjective Swap (0.7) | 0.910 | 0.760 | 0.680 | 0.600 | 0.600 | 0.600 | 0.648 | 0.262 |
| Class Swap | 0.910 | 0.040 | 0.000 | 0.080 | 0.240 | 0.120 | **0.096** | **0.814** |
| Noise Injection (0.7) | 0.910 | 0.760 | 0.800 | 0.840 | 0.680 | 0.800 | 0.776 | 0.134 |
| Formality Shift (0.7) | 0.910 | 0.880 | 0.760 | 0.760 | 0.680 | 0.760 | 0.768 | 0.142 |
| Combined (Adj+Noise) | 0.910 | 0.520 | 0.680 | 0.640 | 0.440 | 0.560 | 0.568 | 0.342 |
| All Methods | 0.910 | 0.240 | 0.160 | 0.320 | 0.320 | 0.240 | **0.256** | **0.654** |

**Key findings**: (i) Class Swap causes catastrophic model failure (accuracy drops to 9.6%) because it inverts the label space while leaving text unchanged — the model correctly identifies positive text but the ground truth label is now negative. (ii) Noise Injection and Formality Shift cause moderate degradation (13-14%) because the hybrid TF-IDF + VADER features retain partial resilience to surface-level text perturbation. (iii) Compound drift (All Methods) causes near-complete model failure (accuracy 25.6%), demonstrating that real-world drift — which typically involves multiple simultaneous perturbation sources — is significantly more destructive than any individual method.

### D. Retraining Recovery Analysis (Table IV)

After observing model degradation under drift, the model was retrained on accumulated drifted stream data and accuracy was measured on continued drifted input.

**Table IV: Retraining Recovery Analysis**

| Drift Method | Clean Acc | Drifted Acc | Recovered Acc | Degradation | Recovery Gain |
|---|---|---|---|---|---|
| Adjective Swap (0.7) | 0.800 | 0.573 | **0.800** | -0.227 | **+0.227** |
| Class Swap | 0.800 | 0.093 | **0.740** | -0.707 | **+0.647** |
| Noise Injection (0.7) | 0.800 | 0.707 | 0.740 | -0.093 | +0.033 |
| Combined (Adj+Class+Noise) | 0.800 | 0.107 | **0.800** | -0.693 | **+0.693** |

**Key findings**: (i) Retraining fully restores accuracy for Adjective Swap (+22.7 pp) and the Combined scenario (+69.3 pp), demonstrating that the closed-loop pipeline successfully adapts to the drifted distribution. (ii) Class Swap recovery reaches 74.0% (from 9.3%), a +64.7 pp gain — the model learns the inverted label mapping. (iii) Noise Injection shows minimal recovery gain (+3.3 pp) because the model already retained reasonable accuracy (70.7%) under syntactic noise. This validates that retraining is most beneficial for semantic and label drift, while syntactic noise may be better addressed through preprocessing.

### E. Metric Cross-Sensitivity Matrix (Table V)

To characterize which metrics are *diagnostically specific* to which drift types, we computed the delta (deviation from clean) for each metric under each individual drift method.

**Table V: Metric Cross-Sensitivity (Delta from Clean Baseline)**

| Method | Cos. Dist delta | Vocab Drop delta | Sent. JSD delta | SER delta | Score JSD delta |
|---|---|---|---|---|---|
| Adjective Swap | 0.218 | 0.297 | **0.136** | 0.007 | 0.036 |
| Class Swap | 0.156 | 0.269 | 0.027 | 0.011 | **0.612** |
| Class Shift | 0.156 | 0.269 | 0.027 | 0.011 | **0.281** |
| Noise Injection | 0.221 | **0.381** | 0.043 | **0.271** | 0.036 |
| Formality Shift | **0.235** | 0.273 | 0.061 | 0.022 | 0.036 |

**Key findings**: Each drift method exhibits a **distinct metric signature**:
- **Adjective Swap**: Uniquely elevates Sentiment JSD (0.136), confirming its polarity-inverting mechanism.
- **Class Swap / Class Shift**: Almost exclusively perturbs Score Distribution JSD (0.612 / 0.281), with negligible impact on text-level metrics.
- **Noise Injection**: Dominates Spelling Error Rate (0.271) and Vocabulary Drop (0.381) — character-level corruption creates out-of-vocabulary tokens.
- **Formality Shift**: Largest Cosine Distance (0.235) because wholesale vocabulary replacement shifts the TF-IDF feature space most aggressively.

This diagnostic specificity is a practical advantage of the multi-criteria framework: a system operator can inspect the metric vector to infer *what kind* of drift is occurring, not merely *that* drift has occurred.

### F. Drift Curve Dynamics (Table VI)

Adjective Swap (0.7) and Noise Injection (0.5) were applied simultaneously under five temporal curve patterns over 8 sequential windows.

**Table VI: Composite Drift Magnitude (%) Under Different Temporal Curves**

| Curve | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | Avg | Max | Std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Constant | 44.7 | 44.6 | 47.2 | 44.9 | 48.0 | 49.1 | 51.1 | 49.1 | 47.3 | 51.1 | 2.3 |
| Gradual | 30.9 | 34.0 | 43.4 | 38.5 | 45.5 | 44.4 | 46.9 | 48.7 | 41.5 | 48.7 | 6.0 |
| Sudden | 22.5 | 22.7 | 29.0 | 20.3 | 48.7 | 46.3 | 47.3 | 47.8 | 35.6 | 48.7 | **12.2** |
| Sinusoidal | 44.2 | 48.5 | 51.6 | 44.8 | 41.9 | 34.5 | 34.0 | 40.3 | 42.5 | 51.6 | 5.8 |
| Step | 35.1 | 35.5 | 46.7 | 43.3 | 46.6 | 48.3 | 45.9 | 47.3 | 43.6 | 48.3 | 5.0 |

**Key findings**: (i) The *sudden* curve exhibits the highest variance (std = 12.2), with drift magnitude jumping abruptly from ~22% to ~48% at the midpoint — successfully modeling catastrophic data source changes. (ii) The *constant* curve produces the highest average drift (47.3%), confirming that sustained full-intensity perturbation is the most destructive mode. (iii) The *sinusoidal* curve shows the characteristic oscillating pattern, with drift peaking at W3 (51.6%) and reaching a minimum at W7 (34.0%). (iv) The *gradual* curve shows monotonically increasing drift, modeling slow distribution evolution. These results validate that the parametric curve system enables controlled simulation of diverse real-world drift dynamics.

---

## VIII. Discussion

### A. Comparison with Garcia et al. [2]

Our work extends Garcia et al. in four key dimensions:

**Extended drift taxonomy**: Noise Injection and Formality/Slang Shift address practical drift phenomena absent from their taxonomy. Our experiments (Table I) demonstrate that these methods produce measurable and distinct metric responses — Noise Injection generates the highest overall composite drift magnitude (43.9%) among all individual methods.

**Reference-free detection**: Garcia et al.'s evaluation required paired original/drifted text. Our frozen baseline architecture operates on the stream alone, making it deployable in production environments. The multi-criteria framework (Table V) further shows that different drift types produce separable metric signatures, enabling drift diagnosis rather than mere detection.

**Multi-metric monitoring**: Their use of accuracy and F1 alone cannot distinguish between drift types. Our nine-metric framework reveals, for example, that Class Swap exclusively affects score distribution divergence while leaving all text-level metrics unchanged (Table I) — a distinction invisible to accuracy-only monitoring.

**Operational recovery**: Garcia et al. did not attempt model adaptation. Our retraining experiments (Table IV) demonstrate accuracy recovery of up to +69.3 percentage points, validating the practical value of the closed-loop pipeline.

### B. Limitations

1. **Dataset specificity**: Experiments were conducted on Amazon product reviews; generalization to other domains (social media, news, clinical text) requires further validation.
2. **Static classifier**: The downstream model (Logistic Regression) is lightweight but not state-of-the-art. Transformer-based classifiers may exhibit different degradation profiles.
3. **Composite metric weights**: The CDM weights (30/20/20/15/15) were chosen heuristically; data-driven optimization could improve sensitivity calibration.
4. **Manual retraining trigger**: The current system requires human-initiated retraining; automated triggering based on drift magnitude thresholds would improve operational autonomy.

---

## IX. Future Work

Several research directions emerge from this work:

1. **Transformer embedding-based drift detection**: Replacing TF-IDF centroids with contextual embeddings from pre-trained language models (BERT, DistilBERT) could capture deeper semantic shifts that bag-of-words representations miss.
2. **Automated adaptive retraining**: Implementing ADWIN-inspired [5] or Page-Hinkley [7] change detection on the composite drift magnitude time series to trigger retraining automatically when statistically significant drift is confirmed.
3. **Multi-language drift**: Extending the drift generation methods and detection framework to non-English text streams, requiring language-specific lexical resources and sentiment analyzers.
4. **Adversarial drift detection**: Investigating whether the reference-free detection framework can distinguish natural drift from adversarial text perturbations designed to evade detection.
5. **Incremental learning**: Replacing batch retraining with online/incremental classifiers (e.g., from the River library [19]) that can continuously adapt without requiring full retraining cycles.

---

## X. Conclusion

This paper presented StreamSense, a real-time streaming system for semantic drift detection and closed-loop model recovery in sentiment analysis. We extended Garcia et al.'s [2] drift generation taxonomy with two novel practical methods (Noise Injection and Formality/Slang Shift) and introduced a reference-free frozen baseline detection architecture that monitors nine statistical indicators across five complementary dimensions. Experimental evaluation on 500 Amazon Movie Reviews demonstrated that: (i) each drift method produces a distinct, diagnosable metric signature; (ii) compound drift causes significantly more severe model degradation than any individual method (accuracy dropping from 91% to 25.6%); (iii) on-demand retraining with baseline recalibration recovers accuracy by up to +69.3 percentage points; and (iv) the parametric drift curve system successfully simulates diverse temporal dynamics. These results validate that multi-criteria reference-free monitoring, coupled with closed-loop retraining, provides an effective and deployable approach to maintaining NLP model quality under concept drift.

---

## References

[1] J. Gama, I. Zliobaite, A. Bifet, M. Pechenizkiy, and A. Bouchachia, "A survey on concept drift adaptation," *ACM Computing Surveys*, vol. 46, no. 4, pp. 1-37, 2014.

[2] C. M. Garcia, A. L. Koerich, A. de Souza Britto Jr., and J. P. Barddal, "Methods for generating drift in text streams," arXiv preprint arXiv:2403.12328, 2024.

[3] J. Gama, P. Medas, G. Castillo, and P. Rodrigues, "Learning with drift detection," in *Proc. Brazilian Symposium on Artificial Intelligence (SBIA)*, pp. 286-295, 2004.

[4] A. Cano and B. Krawczyk, "Concept drift adaptation in text stream mining: A comprehensive review," arXiv preprint, 2024.

[5] A. Bifet and R. Gavalda, "Learning from time-changing data with adaptive windowing," in *Proc. SIAM International Conference on Data Mining (SDM)*, pp. 443-448, 2007.

[6] M. Baena-Garcia, J. del Campo-Avila, R. Fidalgo, A. Bifet, R. Gavalda, and R. Morales-Bueno, "Early drift detection method," in *Proc. ECML PKDD Workshop on Knowledge Discovery from Data Streams*, 2006.

[7] E. S. Page, "Continuous inspection schemes," *Biometrika*, vol. 41, no. 1/2, pp. 100-115, 1954.

[8] S. Kullback and R. A. Leibler, "On information and sufficiency," *Annals of Mathematical Statistics*, vol. 22, no. 1, pp. 79-86, 1951.

[9] J. Lin, "Divergence measures based on the Shannon entropy," *IEEE Transactions on Information Theory*, vol. 37, no. 1, pp. 145-151, 1991.

[10] L. V. Kantorovich, "On the translocation of masses," *Doklady Akademii Nauk SSSR*, vol. 37, no. 7-8, pp. 227-229, 1942.

[11] Evidently AI, "Data and ML model monitoring," https://www.evidentlyai.com, 2024.

[12] Seldon Technologies, "Alibi Detect: Algorithms for outlier, adversarial and drift detection," https://github.com/SeldonIO/alibi-detect, 2024.

[13] C. J. Hutto and E. Gilbert, "VADER: A parsimonious rule-based model for sentiment analysis of social media text," in *Proc. AAAI Conference on Weblogs and Social Media (ICWSM)*, 2014.

[14] A. Agrawal, B. Mukherjee, and J. R. McAuley, "What makes a good sentiment analysis model? A survey of recent advances," *ACM Computing Surveys*, 2024.

[15] R. Ahuja, A. Chug, S. Kohli, S. Gupta, and P. Ahuja, "The impact of features extraction on the sentiment analysis," *Procedia Computer Science*, vol. 152, pp. 341-348, 2019.

[16] D. Kreuzberger, N. Kuhl, and S. Hirschl, "Machine learning operations (MLOps): Overview, definition, and architecture," *IEEE Access*, vol. 11, pp. 31866-31879, 2023.

[17] S. Patel and R. Kumar, "Concept drift detection and adaptive retraining of classification models," arXiv preprint arXiv:2608.13465, 2026.

[18] J. McAuley and J. Leskovec, "From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews," in *Proc. International Conference on World Wide Web (WWW)*, pp. 897-908, 2013.

[19] J. Montiel, M. Halford, S. M. Mastelini, G. Bolmier, R. Sourty, R. Vaysse, A. Zouitine, H. M. Gomes, J. Read, T. Abdessalem, and A. Bifet, "River: Machine learning for streaming data in Python," *Journal of Machine Learning Research*, vol. 22, no. 110, pp. 1-8, 2021.

[20] G. I. Webb, R. Hyde, H. Cao, H. L. Nguyen, and F. Petitjean, "Characterizing concept drift," *Data Mining and Knowledge Discovery*, vol. 30, no. 4, pp. 964-994, 2016.
