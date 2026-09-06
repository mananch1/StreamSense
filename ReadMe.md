# StreamSense ⚡
> **Real-time Social Feed Summarization & Sentiment Analysis under Synthetic Semantic Data Drift**  
> *Advanced Machine Learning (AML) Project — Sem 7*

![Project Architecture](gist.png)

---

## 🌟 Project Highlights & AML Contributions

StreamSense investigates how modern Natural Language Processing (NLP) models perform when deployed on streaming social feeds subject to **concept drift** and **semantic data drift**.

### 1. Dual-Component System Architecture
- **Control System & Synthetic Drift Engine**: A dynamic control panel providing fine-grained parametric control over data streams, drift velocity, and multiple textual corruption/drift strategies.
- **Streaming Social Feed Processor & Evaluator**: Consumes incoming text streams under a dual-trigger windowing strategy ($N$ messages or $T$ seconds timeout, whichever occurs first) and monitors statistical divergence in real-time.

### 2. Academically Grounded Drift Methodologies
Incorporates drift generation paradigms from [Garcia et al. (2024), *Methods for Generating Drift in Text Streams*, arXiv:2403.12328](https://arxiv.org/abs/2403.12328), combined with custom NLP perturbations:

| Drift Method | Taxonomy | Description | Practical Real-World Phenomenon |
|---|---|---|---|
| **Adjective Swap** | Semantic / WordNet | Tokenizes text, POS-tags adjectives (`JJ`, `JJR`, `JJS`), and swaps them with WordNet antonyms. | Semantic shift & opinion reversal |
| **Class Swap** | Abrupt Label Drift | Inverts rating scales ($1\star \leftrightarrow 5\star$, $2\star \leftrightarrow 4\star$, $3\star \text{ unchanged}$). | Sudden platform guideline recalibration |
| **Class Shift** | Gradual Label Drift | Cyclically shifts rating classes ($1 \to 2 \to 3 \to 4 \to 5 \to 1$). | Gradual score inflation / drift |
| **Formality & Slang Shift** | Register Transformation | Replaces formal vocabulary with internet slang, abbreviations, and informal phrasing. | Generational/demographic community shifts |
| **Noise & Typo Injection** | Syntactic Perturbation | Simulates keyboard adjacency typos, missing characters, transpositions, and truncation. | Bot traffic, noisy mobile keyboards |
| **Time-Slice Removal** | Temporal Discontinuity | Simulates missing historical windows in stream mining. | System outages, missing data epochs |

### 3. Real-Time Drift Metrics & Statistical Formulations

StreamSense operates in a **reference-free, baseline-windowed** mode where the metrics engine monitors incoming stream windows directly against a frozen reference profile established during an initial burn-in period:

| Metric | Domain / Range | Mathematical Formulation | Statistical Role & Interpretation |
|---|---|---|---|
| **Centroid TF-IDF Cosine Similarity** | $[0.0, 1.0]$ | $\text{Sim}_{\text{centroid}} = \frac{\boldsymbol{\mu}_{\text{baseline}} \cdot \boldsymbol{\mu}_{\text{curr}}}{\|\boldsymbol{\mu}_{\text{baseline}}\|_2 \|\boldsymbol{\mu}_{\text{curr}}\|_2}$ | Measures angular divergence between the frozen baseline centroid $\boldsymbol{\mu}_{\text{baseline}}$ and the current window centroid $\boldsymbol{\mu}_{\text{curr}}$. Values $< 1.0$ indicate semantic vocabulary displacement. |
| **Vocabulary Jaccard Overlap** | $[0.0, 1.0]$ | $J(V_{\text{baseline}}, V_{\text{curr}}) = \frac{\|V_{\text{baseline}} \cap V_{\text{curr}}\|}{\|V_{\text{baseline}} \cup V_{\text{curr}}\|}$ | Quantifies lexical retention vs Out-of-Vocabulary (OOV) expansion against the union of all vocabulary observed during burn-in. |
| **Sentiment Polarity Distribution** | $\Delta^2$ Simplex | $P(c) = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\text{VADER}(t_i) \in c)$ | Categorical sentiment distribution across classes $c \in \{\text{positive}, \text{neutral}, \text{negative}\}$ using VADER compound polarity thresholds ($\ge 0.05, \le -0.05$). |
| **Kullback-Leibler (KL) Divergence** | $[0.0, +\infty)$ | $D_{\text{KL}}(P_{\text{baseline}} \parallel Q_{\text{curr}}) = \sum_{c} P(c) \ln \frac{P(c)}{Q(c)}$ | Information-theoretic relative entropy from the baseline sentiment profile $P_{\text{baseline}}$ to the current window distribution $Q_{\text{curr}}$ (with epsilon smoothing). |
| **Jensen-Shannon Divergence (JSD)** | $[0.0, 1.0]$ | $\text{JSD}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$ | Symmetric, bounded divergence where $M = \frac{1}{2}(P + Q)$. Its square root $\sqrt{\text{JSD}}$ is a true metric satisfying the triangle inequality. |
| **Sentiment Wasserstein-1 Distance** | $[0.0, 2.0]$ | $W_1(P, Q) = \sum_{k} \|F_P(k) - F_Q(k)\|$ | Earth Mover's Distance over ordered sentiment states ($\text{neg} < \text{neu} < \text{pos}$). Penalizes polarity reversals ($\text{pos} \leftrightarrow \text{neg}$) twice as heavily as neutrality shifts. |
| **Spelling Error Rate (SER)** | $[0.0, 1.0]$ | $\text{SER} = \frac{1}{N_{\text{tokens}}}\sum_{j=1}^{N_{\text{tokens}}} \mathbb{I}(w_j \notin \text{Vocab}_{\text{en}})$ | Reference-free syntactic noise measurement via English dictionary lookup without requiring paired original text. |
| **Score Distribution Divergence** | $[0.0, 1.0]$ | $\text{JSD}(S_{\text{baseline}} \parallel S_{\text{curr}})$ | Jensen-Shannon Divergence over 5-bin histogram of star ratings ($1.0\star - 5.0\star$) detecting rating drift and label inversion. |
| **Composite Drift Magnitude (%)** | $[0.0\%, 100.0\%]$ | $\text{Score} = (0.30 d_{\cos} + 0.20 d_{\text{vocab}} + 0.20 \sqrt{\text{JSD}} + 0.15 \Delta_{\text{SER}} + 0.15 \text{JSD}_{\text{score}}) \times 100$ | Multi-criteria unified perturbation magnitude synthesizing semantic, lexical, sentiment, syntactic, and score shifts into an interpretable percentage. |

> **Note on Deprecated Paired Metrics**: Pairwise Cosine Similarity, Character Mutation Rate (CER via Levenshtein edit distance), and Average Score Delta (MAE) are deprecated and removed from active monitoring because they require clean parallel reference text, which is unavailable in real-world streaming deployments. They are retained in the codebase for offline benchmarking and ablation studies.

### 4. Reference-Free Detection Architecture
- **Burn-In Calibration**: The engine establishes a reference baseline profile across $B$ clean initial windows (default: 3 windows, configurable 1–10 via the control panel).
- **Frozen Baseline Strategy**: Once calibrated, the baseline profile is permanently frozen. It is **never** updated with subsequent windows, completely avoiding the "boiling frog" vulnerability where gradual semantic drift adapts the baseline unnoticed.
- **Side-by-Side Educational UI**: The frontend maintains a side-by-side stream comparison (Original vs Drifted) for educational and interactive demonstration purposes, while the underlying metrics engine operates reference-free.

### 5. Closed-Loop MLOps Pipeline: Observability & Retraining
StreamSense closes the operational machine learning loop by tying unsupervised drift detection directly to supervised downstream model performance:
1. **Warm-Up Training (`Model v1.0`)**: Once the clean burn-in baseline finishes, a sentiment classifier (`TfidfVectorizer` + `LogisticRegression`) is automatically trained on the initial clean reviews.
2. **Streaming Inference & Observability**: Every incoming streamed review is evaluated by the active model. Its predicted sentiment is compared against the review's star rating proxy label (`4-5★` Positive, `3★` Neutral, `1-2★` Negative) to calculate real-time window and cumulative accuracy.
3. **Drift Degradation Feedback**: When semantic perturbations (adjective antonym swaps, class swaps, or typographical noise) are enabled, the dashboard displays accuracy plunging from $>90\%$ to $<45\%$, visually plotted alongside rising Composite Drift Magnitude.
4. **On-Demand Retraining (`Model v2.0+`)**: The user can trigger retraining directly from the Control Panel at any time. The engine trains an updated classifier on recent accumulated stream samples, restoring operational accuracy on the drifted distribution.

---

## 📊 Dataset: Stanford SNAP Amazon Movie Reviews
- **Source**: [Stanford SNAP Amazon Movie Reviews](https://snap.stanford.edu/data/web-Movies.html) (~8 million reviews)
- **Top Product Extracted**: `B002QZ1RS6` (*Beachbody INSANITY Series* — 957 detailed long-form reviews)
- **Stream Visibility**: Review items streamed through the feed simulator include both fields for dashboard display, but the metrics engine strictly inspects only the streamed/drifted text and score.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-extract Dataset Subset
```bash
python scripts/create_subset.py
```

### 3. Run Unit Tests
```bash
python -m unittest discover tests -v
```

### 4. Launch StreamSense
```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to access the interactive Control System & Real-Time Analytics Dashboard.

---

## 📁 Repository Structure
```
Project/
├── control_panel/              # Frontend Dashboard
│   ├── index.html              # Dark glassmorphic interface
│   ├── style.css               # Modern styling & animations
│   └── app.js                  # WebSocket client & Chart.js visualizations
├── data/
│   ├── dataset_meta.json       # Extracted product metadata & score distribution
│   ├── sample_reviews.csv      # Rapid development sample (500 reviews)
│   └── top_movie_reviews.csv   # Full review stream for top movie (957 reviews)
├── scripts/
│   └── create_subset.py        # Ultra-fast, memory-safe two-pass 9.3GB stream parser
├── src/
│   ├── config.py               # Pydantic schemas & configuration
│   ├── data_loader.py          # Sequential review stream manager
│   ├── drift_engine.py         # 6 Textual & semantic drift generator methods
│   ├── drift_metrics.py        # Real-time cosine, KL-div, and overlap calculators
│   ├── feed_simulator.py       # Dual-trigger async window streaming simulator
│   └── main.py                 # FastAPI backend & WebSocket endpoints
├── tests/
│   └── test_drift_engine.py    # Automated test suite
├── requirements.txt            # Python dependencies
└── ReadMe.md                   # Project documentation
```

---

## 📚 References
- Garcia, C. M., Koerich, A. L., de Souza Britto Jr, A., & Barddal, J. P. (2024). *Methods for Generating Drift in Text Streams*. [arXiv:2403.12328](https://arxiv.org/abs/2403.12328).
- McAuley, J., & Leskovec, J. (2013). *From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews*. WWW 2013.