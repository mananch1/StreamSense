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

StreamSense implements multi-layer drift detection spanning syntactic, semantic, lexical, and distributional divergence across tumbling/sliding stream windows:

| Metric | Domain / Range | Mathematical Formulation | Statistical Role & Interpretation |
|---|---|---|---|
| **Centroid TF-IDF Cosine Similarity** | $[0.0, 1.0]$ | $\text{Sim}_{\text{centroid}} = \frac{\boldsymbol{\mu}_{\text{orig}} \cdot \boldsymbol{\mu}_{\text{drift}}}{\|\boldsymbol{\mu}_{\text{orig}}\|_2 \|\boldsymbol{\mu}_{\text{drift}}\|_2}$ | Measures angular divergence between window mean TF-IDF centroids $\boldsymbol{\mu} = \frac{1}{N}\sum \mathbf{x}_i$. Values $< 1.0$ indicate semantic term displacement. |
| **Pairwise Cosine Similarity** | $[0.0, 1.0]$ | $\text{Sim}_{\text{pair}} = \frac{1}{N}\sum_{i=1}^N \frac{\mathbf{x}_i \cdot \mathbf{x}'_i}{\|\mathbf{x}_i\|_2 \|\mathbf{x}'_i\|_2}$ | Measures instance-by-instance perturbation severity between paired original review $\mathbf{x}_i$ and drifted review $\mathbf{x}'_i$. |
| **Vocabulary Jaccard Overlap** | $[0.0, 1.0]$ | $J(V_{\text{orig}}, V_{\text{drift}}) = \frac{\|V_{\text{orig}} \cap V_{\text{drift}}\|}{\|V_{\text{orig}} \cup V_{\text{drift}}\|}$ | Quantifies lexical retention vs Out-of-Vocabulary (OOV) expansion across alphabetic token sets $V$. |
| **Sentiment Polarity Distribution** | $\Delta^2$ Simplex | $P(c) = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\text{VADER}(t_i) \in c)$ | Categorical sentiment distribution across classes $c \in \{\text{positive}, \text{neutral}, \text{negative}\}$ using VADER compound polarity thresholds ($\ge 0.05, \le -0.05$). |
| **Kullback-Leibler (KL) Divergence** | $[0.0, +\infty)$ | $D_{\text{KL}}(P \parallel Q) = \sum_{c} P(c) \ln \frac{P(c)}{Q(c)}$ | Information-theoretic relative entropy from reference distribution $P$ to drifted distribution $Q$ (with Laplace/epsilon smoothing). |
| **Jensen-Shannon Divergence (JSD)** | $[0.0, 1.0]$ | $\text{JSD}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$ | Symmetric, strictly bounded divergence where $M = \frac{1}{2}(P + Q)$. Its square root $\sqrt{\text{JSD}}$ is a true metric satisfying the triangle inequality. |
| **Sentiment Wasserstein-1 Distance** | $[0.0, 2.0]$ | $W_1(P, Q) = \sum_{k} \|F_P(k) - F_Q(k)\|$ | Earth Mover's Distance over ordered sentiment states ($\text{neg} < \text{neu} < \text{pos}$). Penalizes polarity reversals ($\text{pos} \leftrightarrow \text{neg}$) twice as heavily as neutrality shifts ($\text{pos} \leftrightarrow \text{neu}$). |
| **Character Mutation Rate (CER)** | $[0.0, 1.0]$ | $\text{CER} = \frac{1}{N}\sum_{i=1}^N \frac{\text{Levenshtein}(t_i, t'_i)}{\max(\|t_i\|, 1)}$ | Normalized character edit distance capturing typographical noise, transposition, and character injection. |
| **Average Score Delta ($\text{MAE}_{\text{score}}$)** | $[0.0, 4.0]$ | $\Delta \bar{s} = \frac{1}{N}\sum_{i=1}^N \|s_i^{\text{drift}} - s_i^{\text{orig}}\|$ | Mean absolute error on star ratings ($1\star - 5\star$) reflecting label degradation. |
| **Composite Drift Magnitude (%)** | $[0.0\%, 100.0\%]$ | $\text{Score} = (0.35 d_{\cos} + 0.25 d_{\text{vocab}} + 0.20 \sqrt{\text{JSD}} + 0.20 \frac{\Delta \bar{s}}{4.0}) \times 100$ | Multi-criteria unified perturbation magnitude score synthesizing lexical, semantic, sentiment, and label shifts into an interpretable percentage. |

---

## 📊 Dataset: Stanford SNAP Amazon Movie Reviews
- **Source**: [Stanford SNAP Amazon Movie Reviews](https://snap.stanford.edu/data/web-Movies.html) (~8 million reviews)
- **Top Product Extracted**: `B002QZ1RS6` (*Beachbody INSANITY Series* — 957 detailed long-form reviews)
- **Hidden Ground Truth**: Review star ratings ($1.0 - 5.0\star$) are stripped from the downstream consumer feed and preserved internally to measure accuracy degradation over time.

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
python -m unittest tests/test_drift_engine.py
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