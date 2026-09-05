"""
StreamSense Drift Metrics.
=========================
Calculates statistical, semantic, and syntactic data drift indicators
using a reference-free, baseline-windowed streaming architecture.

Academic & Contemporary Industry Foundation:
--------------------------------------------
1. Garcia et al. (2024), "Methods for Generating Drift in Text Streams", arXiv:2403.12328
2. Hutto & Gilbert (2014), "VADER: A Parsimonious Rule-based Model for Sentiment Analysis", ICWSM
3. Evidently AI & Seldon Alibi-Detect: Contemporary streaming NLP drift detection standards
   (Jensen-Shannon Divergence, Earth Mover's / Wasserstein Distance, Vector Space Cosine Distance,
   Spelling Error Rates, and Categorical Label Shift)

Reference-Free Architecture:
----------------------------
In real-world deployment, clean parallel ground-truth text is not available at inference time.
StreamSense establishes a reference profile during an initial "burn-in" period of N clean windows.
Once calibrated, the baseline profile is frozen permanently (preventing the "boiling frog" failure
mode where gradual drift adapts the baseline unnoticed). Subsequent streaming windows are compared
directly against this frozen baseline profile.

Active Monitored Metrics (Reference-Free):
------------------------------------------
1. TF-IDF Centroid Cosine Similarity [0.0, 1.0] (vs Baseline Centroid)
2. Vocabulary Jaccard Overlap [0.0, 1.0] (vs Baseline Vocabulary Set)
3. VADER Sentiment Categorical Distribution (Positive / Neutral / Negative)
4. Sentiment Kullback-Leibler (KL) Divergence [0.0, +inf) with epsilon smoothing
5. Sentiment Jensen-Shannon Divergence (JSD) [0.0, 1.0] (Symmetric, bounded relative entropy)
6. Sentiment Wasserstein-1 Distance (Earth Mover's Distance) [0.0, 2.0] for ordinal sentiment
7. Spelling Error Rate [0.0, 1.0] (Dictionary-based token verification)
8. Rating Score Distribution Divergence (JSD over 1.0-5.0 star histogram) [0.0, 1.0]
9. Composite Drift Magnitude Score [0.0%, 100.0%] (Multi-criteria weighted perturbation index)

Deprecated Metrics (Retained with Documentation):
-------------------------------------------------
- Pairwise Cosine Similarity: Requires paired (X_i, X'_i) texts.
- Character Error Rate (CER): Requires paired original and corrupted strings.
- Average Rating Score Delta (MAE): Requires paired original and shifted scores.
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk import edit_distance

vader_analyzer = SentimentIntensityAnalyzer()

# Load English dictionary for Spelling Error Rate
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)
from nltk.corpus import words as nltk_words
ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())


class DriftMetricsCalculator:
    """
    Computes statistical and semantic data drift indicators across streaming
    windows using a reference-free, baseline-windowed methodology.
    """

    def __init__(self, burn_in_windows: int = 3):
        self.history: List[Dict[str, Any]] = []

        # Burn-in baseline state
        self.burn_in_windows_needed: int = max(1, burn_in_windows)
        self.burn_in_windows_collected: int = 0
        self.baseline_ready: bool = False

        # Accumulated baseline data (collected during burn-in, frozen after)
        self._baseline_texts: List[str] = []
        self._baseline_vocab: set = set()
        self._baseline_sentiment_counts: Dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
        self._baseline_sentiment_total: int = 0
        self._baseline_sentiment_dist: Dict[str, float] = {}
        self._baseline_score_counts: Dict[str, int] = {}
        self._baseline_score_total: int = 0
        self._baseline_score_dist: Dict[str, float] = {}
        self._baseline_spelling_error_rate: float = 0.0
        self._baseline_spelling_error_rates: List[float] = []

        # TF-IDF baseline (fitted during burn-in, frozen after)
        self._baseline_vectorizer: Optional[TfidfVectorizer] = None
        self._baseline_centroid: Optional[np.ndarray] = None

    def reset_baseline(self):
        """Clear baseline and re-enter burn-in phase."""
        self.burn_in_windows_collected = 0
        self.baseline_ready = False
        self._baseline_texts = []
        self._baseline_vocab = set()
        self._baseline_sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        self._baseline_sentiment_total = 0
        self._baseline_sentiment_dist = {}
        self._baseline_score_counts = {}
        self._baseline_score_total = 0
        self._baseline_score_dist = {}
        self._baseline_spelling_error_rate = 0.0
        self._baseline_spelling_error_rates = []
        self._baseline_vectorizer = None
        self._baseline_centroid = None
        self.history = []

    def compute_window_metrics(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reference-free drift detection across a streaming window of reviews.

        During burn-in: Accumulates statistics into baseline profile.
        After burn-in: Compares current window statistics against frozen baseline.

        Args:
            items: List of review dicts. Only uses 'drifted_text' and 'drifted_score'
                   fields (the stream-visible data). Does NOT use 'original_text' or
                   'original_score' — those exist in the dict for UI display only.

        Returns:
            Dict[str, Any] containing all statistical metrics and telemetry records.
        """
        if not items:
            return self._empty_metrics()

        # Extract only stream-visible data (reference-free)
        stream_texts = [it.get("drifted_text", "") for it in items]
        stream_scores = [float(it.get("drifted_score", 3.0)) for it in items]
        drifted_count = sum(1 for it in items if it.get("is_drifted", False))
        drift_prop = drifted_count / len(items)

        # ----------------------------------------------------
        # PHASE 1: BURN-IN CALIBRATION
        # ----------------------------------------------------
        if not self.baseline_ready:
            # 1. Accumulate raw texts
            self._baseline_texts.extend(stream_texts)

            # 2. Accumulate vocabulary
            for t in stream_texts:
                tokens = re.findall(r'\b[a-zA-Z]{3,}\b', t.lower())
                self._baseline_vocab.update(tokens)

            # 3. Accumulate sentiment counts
            for t in stream_texts:
                if not t.strip():
                    self._baseline_sentiment_counts["neutral"] += 1
                else:
                    vs = vader_analyzer.polarity_scores(t)
                    compound = vs['compound']
                    if compound >= 0.05:
                        self._baseline_sentiment_counts["positive"] += 1
                    elif compound <= -0.05:
                        self._baseline_sentiment_counts["negative"] += 1
                    else:
                        self._baseline_sentiment_counts["neutral"] += 1
                self._baseline_sentiment_total += 1

            # 4. Accumulate score counts
            for s in stream_scores:
                key = str(float(min(5.0, max(1.0, round(s)))))
                self._baseline_score_counts[key] = self._baseline_score_counts.get(key, 0) + 1
                self._baseline_score_total += 1

            # 5. Accumulate spelling error rate
            window_ser = self._compute_spelling_error_rate(stream_texts)
            self._baseline_spelling_error_rates.append(window_ser)

            self.burn_in_windows_collected += 1

            # Check if burn-in phase is complete
            if self.burn_in_windows_collected >= self.burn_in_windows_needed:
                self._finalize_baseline()

            current_sent_dist = self._get_sentiment_distribution(stream_texts)
            current_score_dist = self._compute_score_distribution(stream_scores)

            burn_in_record = {
                "phase": "burn_in",
                "burn_in_progress": f"{self.burn_in_windows_collected}/{self.burn_in_windows_needed}",
                "window_size": len(items),
                "cosine_similarity": 1.0,
                "vocab_overlap": 1.0,
                "sentiment_kl_divergence": 0.0,
                "sentiment_js_divergence": 0.0,
                "sentiment_wasserstein_distance": 0.0,
                "spelling_error_rate": round(float(window_ser), 4),
                "baseline_spelling_error_rate": round(float(self._baseline_spelling_error_rate), 4),
                "current_sentiment_dist": current_sent_dist,
                "baseline_sentiment_dist": dict(self._baseline_sentiment_dist) if self.baseline_ready else current_sent_dist,
                "current_score_dist": current_score_dist,
                "baseline_score_dist": dict(self._baseline_score_dist) if self.baseline_ready else current_score_dist,
                "score_dist_divergence": 0.0,
                "drift_magnitude_pct": 0.0,
                "drifted_items_count": drifted_count,
                "drifted_items_ratio": round(float(drift_prop), 3),
                # Deprecated compatibility keys
                "pairwise_cosine_similarity": 1.0,
                "character_mutation_rate": 0.0,
                "avg_score_delta": 0.0,
                "original_sentiment_dist": dict(self._baseline_sentiment_dist) if self.baseline_ready else current_sent_dist,
                "drifted_sentiment_dist": current_sent_dist,
            }

            self._record_history(burn_in_record)
            return burn_in_record

        # ----------------------------------------------------
        # PHASE 2: MONITORING AGAINST FROZEN BASELINE
        # ----------------------------------------------------
        # 1. TF-IDF Centroid Cosine Similarity against frozen baseline centroid
        centroid_cosine_sim = self._compute_cosine_similarity(stream_texts)

        # 2. Vocabulary Jaccard Overlap against frozen baseline vocabulary
        vocab_overlap = self._compute_vocab_overlap(stream_texts)

        # 3. Sentiment Distribution Shift against frozen baseline distribution
        current_sent_dist = self._get_sentiment_distribution(stream_texts)
        kl_div = self._compute_kl_divergence(self._baseline_sentiment_dist, current_sent_dist)
        js_div = self._compute_js_divergence(self._baseline_sentiment_dist, current_sent_dist)
        wasserstein_dist = self._compute_sentiment_wasserstein(self._baseline_sentiment_dist, current_sent_dist)

        # 4. Spelling Error Rate via dictionary token verification
        current_spelling_rate = self._compute_spelling_error_rate(stream_texts)

        # 5. Rating Score Distribution Divergence against frozen baseline score distribution
        current_score_dist = self._compute_score_distribution(stream_scores)
        score_dist_divergence = self._compute_score_dist_divergence(self._baseline_score_dist, current_score_dist)

        # 6. Composite Drift Magnitude Score [0.0%, 100.0%]
        # Multi-criteria weighted perturbation index:
        # - Cosine distance: 30%
        # - Vocabulary drop: 20%
        # - Sentiment Jensen-Shannon distance: 20%
        # - Spelling error rate increase: 15%
        # - Score distribution divergence: 15%
        cosine_dist = max(0.0, 1.0 - centroid_cosine_sim)
        vocab_dist = max(0.0, 1.0 - vocab_overlap)
        norm_sent_divergence = math.sqrt(js_div)  # strictly bounded in [0.0, 1.0]

        baseline_ser = max(0.01, self._baseline_spelling_error_rate)
        spelling_delta = min(1.0, max(0.0, current_spelling_rate - self._baseline_spelling_error_rate) / baseline_ser)
        score_div = score_dist_divergence  # strictly bounded in [0.0, 1.0]

        composite_drift = (
            0.30 * cosine_dist +
            0.20 * vocab_dist +
            0.20 * norm_sent_divergence +
            0.15 * spelling_delta +
            0.15 * score_div
        ) * 100.0

        metrics_record = {
            "phase": "monitoring",
            "window_size": len(items),
            "cosine_similarity": round(float(centroid_cosine_sim), 4),
            "vocab_overlap": round(float(vocab_overlap), 4),
            "sentiment_kl_divergence": round(float(kl_div), 4),
            "sentiment_js_divergence": round(float(js_div), 4),
            "sentiment_wasserstein_distance": round(float(wasserstein_dist), 4),
            "spelling_error_rate": round(float(current_spelling_rate), 4),
            "baseline_spelling_error_rate": round(float(self._baseline_spelling_error_rate), 4),
            "current_sentiment_dist": current_sent_dist,
            "baseline_sentiment_dist": dict(self._baseline_sentiment_dist),
            "current_score_dist": current_score_dist,
            "baseline_score_dist": dict(self._baseline_score_dist),
            "score_dist_divergence": round(float(score_dist_divergence), 4),
            "drift_magnitude_pct": round(float(composite_drift), 1),
            "drifted_items_count": drifted_count,
            "drifted_items_ratio": round(float(drift_prop), 3),
            # Deprecated compatibility keys (for backward compatibility with legacy consumers)
            "pairwise_cosine_similarity": round(float(centroid_cosine_sim), 4),
            "character_mutation_rate": round(float(current_spelling_rate), 4),
            "avg_score_delta": round(float(score_dist_divergence * 4.0), 2),
            "original_sentiment_dist": dict(self._baseline_sentiment_dist),
            "drifted_sentiment_dist": current_sent_dist,
        }

        self._record_history(metrics_record)
        return metrics_record

    def _finalize_baseline(self):
        """Finalizes and freezes the baseline reference profile after burn-in."""
        tot_sent = max(1, self._baseline_sentiment_total)
        self._baseline_sentiment_dist = {
            k: round(self._baseline_sentiment_counts.get(k, 0) / tot_sent, 4)
            for k in ["positive", "neutral", "negative"]
        }

        tot_score = max(1, self._baseline_score_total)
        self._baseline_score_dist = {
            k: round(self._baseline_score_counts.get(k, 0) / tot_score, 4)
            for k in ["1.0", "2.0", "3.0", "4.0", "5.0"]
        }

        try:
            self._baseline_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
            tfidf_mat = self._baseline_vectorizer.fit_transform(self._baseline_texts)
            self._baseline_centroid = np.asarray(tfidf_mat.mean(axis=0))
        except Exception:
            self._baseline_vectorizer = None
            self._baseline_centroid = None

        self._baseline_spelling_error_rate = (
            float(np.mean(self._baseline_spelling_error_rates))
            if self._baseline_spelling_error_rates
            else 0.0
        )

        self.baseline_ready = True

    def _empty_metrics(self) -> Dict[str, Any]:
        """Returns safe default metrics for empty streaming windows."""
        phase = "monitoring" if self.baseline_ready else "burn_in"
        progress = f"{self.burn_in_windows_collected}/{self.burn_in_windows_needed}" if not self.baseline_ready else "ready"
        return {
            "phase": phase,
            "burn_in_progress": progress,
            "window_size": 0,
            "cosine_similarity": 1.0,
            "vocab_overlap": 1.0,
            "sentiment_kl_divergence": 0.0,
            "sentiment_js_divergence": 0.0,
            "sentiment_wasserstein_distance": 0.0,
            "spelling_error_rate": 0.0,
            "baseline_spelling_error_rate": round(float(self._baseline_spelling_error_rate), 4),
            "current_sentiment_dist": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
            "baseline_sentiment_dist": dict(self._baseline_sentiment_dist) if self.baseline_ready else {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
            "current_score_dist": {"1.0": 0.0, "2.0": 0.0, "3.0": 0.0, "4.0": 0.0, "5.0": 0.0},
            "baseline_score_dist": dict(self._baseline_score_dist) if self.baseline_ready else {"1.0": 0.0, "2.0": 0.0, "3.0": 0.0, "4.0": 0.0, "5.0": 0.0},
            "score_dist_divergence": 0.0,
            "drift_magnitude_pct": 0.0,
            "drifted_items_count": 0,
            "drifted_items_ratio": 0.0,
            "pairwise_cosine_similarity": 1.0,
            "character_mutation_rate": 0.0,
            "avg_score_delta": 0.0,
            "original_sentiment_dist": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
            "drifted_sentiment_dist": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
        }

    def _record_history(self, record: Dict[str, Any]):
        self.history.append(record)
        if len(self.history) > 100:
            self.history.pop(0)

    def _compute_cosine_similarity(self, texts: List[str]) -> float:
        """
        Computes TF-IDF centroid cosine similarity between the current window
        and the frozen baseline centroid.

        Reference-free: Only current stream texts are passed; they are transformed
        using the fitted baseline vectorizer and compared against the baseline centroid.
        Range: [0.0, 1.0].
        """
        if self._baseline_vectorizer is None or self._baseline_centroid is None or not texts:
            return 1.0
        try:
            tfidf_mat = self._baseline_vectorizer.transform(texts)
            current_centroid = np.asarray(tfidf_mat.mean(axis=0))

            norm_base = np.linalg.norm(self._baseline_centroid)
            norm_curr = np.linalg.norm(current_centroid)
            if norm_base == 0 or norm_curr == 0:
                return 0.0

            sim = float(cosine_similarity(self._baseline_centroid, current_centroid)[0][0])
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 1.0

    def _compute_vocab_overlap(self, texts: List[str]) -> float:
        """
        Computes vocabulary Jaccard similarity index J(Baseline, Current) = |B ∩ C| / |B ∪ C|.
        Reference-free: compares current window vocabulary against the frozen baseline vocabulary.
        Range: [0.0, 1.0].
        """
        current_words = set()
        for t in texts:
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', t.lower())
            current_words.update(tokens)

        if not self._baseline_vocab or not current_words:
            return 1.0

        intersection = len(self._baseline_vocab.intersection(current_words))
        union = len(self._baseline_vocab.union(current_words))
        return float(intersection / union) if union > 0 else 1.0

    def _get_sentiment_distribution(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculates empirical categorical probability distribution P = [P(pos), P(neu), P(neg)]
        using VADER compound sentiment polarity score:
          - Positive: compound >= 0.05
          - Neutral: -0.05 < compound < 0.05
          - Negative: compound <= -0.05
        """
        pos_cnt, neu_cnt, neg_cnt = 0, 0, 0
        for t in texts:
            if not t.strip():
                neu_cnt += 1
                continue
            vs = vader_analyzer.polarity_scores(t)
            compound = vs['compound']
            if compound >= 0.05:
                pos_cnt += 1
            elif compound <= -0.05:
                neg_cnt += 1
            else:
                neu_cnt += 1

        total = max(1, len(texts))
        return {
            "positive": round(pos_cnt / total, 4),
            "neutral": round(neu_cnt / total, 4),
            "negative": round(neg_cnt / total, 4)
        }

    def _compute_kl_divergence(self, p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
        """
        Kullback-Leibler Divergence: D_KL(P || Q) = sum P(x) * ln(P(x) / Q(x))
        with epsilon smoothing to handle zero-probability bins.
        Range: [0.0, +inf).
        """
        eps = 1e-5
        kl = 0.0
        for key in ["positive", "neutral", "negative"]:
            p = max(eps, p_dist.get(key, 0.0))
            q = max(eps, q_dist.get(key, 0.0))
            kl += p * math.log(p / q)
        return max(0.0, kl)

    def _compute_js_divergence(self, p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
        """
        Jensen-Shannon Divergence (JSD):
            M = 0.5 * (P + Q)
            JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
        Using base-2 logarithm guarantees JSD is strictly bounded in [0.0, 1.0].
        """
        keys = ["positive", "neutral", "negative"]
        p = np.array([p_dist.get(k, 0.0) for k in keys], dtype=float)
        q = np.array([q_dist.get(k, 0.0) for k in keys], dtype=float)

        sum_p = p.sum()
        sum_q = q.sum()
        if sum_p == 0 or sum_q == 0:
            return 0.0

        p = p / sum_p
        q = q / sum_q
        m = 0.5 * (p + q)

        def _kl_base2(a, b):
            mask = (a > 0) & (b > 0)
            return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

        jsd = 0.5 * _kl_base2(p, m) + 0.5 * _kl_base2(q, m)
        return float(np.clip(jsd, 0.0, 1.0))

    def _compute_sentiment_wasserstein(
        self, p_dist: Dict[str, float], q_dist: Dict[str, float]
    ) -> float:
        """
        Computes Wasserstein-1 Distance (Earth Mover's Distance) on ordinal sentiment classes.
        Naturally accounts for semantic ordering: Negative (0) < Neutral (1) < Positive (2).
        Range: [0.0, 2.0].
        """
        u_values = [0.0, 1.0, 2.0]
        u_weights = [
            p_dist.get("negative", 0.0),
            p_dist.get("neutral", 0.0),
            p_dist.get("positive", 0.0)
        ]
        v_weights = [
            q_dist.get("negative", 0.0),
            q_dist.get("neutral", 0.0),
            q_dist.get("positive", 0.0)
        ]

        if sum(u_weights) == 0 or sum(v_weights) == 0:
            return 0.0

        w_dist = wasserstein_distance(u_values, u_values, u_weights, v_weights)
        return float(w_dist)

    def _compute_spelling_error_rate(self, texts: List[str]) -> float:
        """
        Computes the fraction of alphabetic tokens (length >= 3) NOT found in the English dictionary.
        Reference-free: Does not require paired original text.
        Range: [0.0, 1.0].
        """
        total_tokens = 0
        misspelled_tokens = 0
        for text in texts:
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            for token in tokens:
                total_tokens += 1
                if token not in ENGLISH_WORDS:
                    misspelled_tokens += 1
        if total_tokens == 0:
            return 0.0
        return float(misspelled_tokens / total_tokens)

    def _compute_score_distribution(self, scores: List[float]) -> Dict[str, float]:
        """
        Computes a 5-bin histogram distribution over star ratings 1.0 to 5.0.
        """
        bins = {"1.0": 0, "2.0": 0, "3.0": 0, "4.0": 0, "5.0": 0}
        for s in scores:
            key = str(float(min(5.0, max(1.0, round(s)))))
            bins[key] = bins.get(key, 0) + 1
        total = max(1, len(scores))
        return {k: round(v / total, 4) for k, v in bins.items()}

    def _compute_score_dist_divergence(
        self, baseline_dist: Dict[str, float], current_dist: Dict[str, float]
    ) -> float:
        """
        Jensen-Shannon Divergence between two score histograms over star ratings 1.0 to 5.0.
        Range: [0.0, 1.0].
        """
        keys = ["1.0", "2.0", "3.0", "4.0", "5.0"]
        p = np.array([baseline_dist.get(k, 0.0) for k in keys], dtype=float)
        q = np.array([current_dist.get(k, 0.0) for k in keys], dtype=float)

        sum_p, sum_q = p.sum(), q.sum()
        if sum_p == 0 or sum_q == 0:
            return 0.0

        p = p / sum_p
        q = q / sum_q
        m = 0.5 * (p + q)

        def _kl2(a, b):
            mask = (a > 0) & (b > 0)
            return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

        jsd = 0.5 * _kl2(p, m) + 0.5 * _kl2(q, m)
        return float(np.clip(jsd, 0.0, 1.0))

    # =========================================================================
    # DEPRECATED METHODS (Kept for offline research, evaluation & benchmarks)
    # =========================================================================

    def _compute_character_mutation_rate(
        self, orig_texts: List[str], drift_texts: List[str]
    ) -> float:
        """
        [DEPRECATED — Reference-Free Migration]
        This method computed Character Error Rate (CER) using normalized Levenshtein edit distance
        between paired original and drifted texts. It was removed from the active pipeline because
        it requires access to the clean original text, which is unavailable in reference-free mode.

        Replaced by: _compute_spelling_error_rate() which uses dictionary lookup on stream text alone.
        Retained for: Offline post-hoc evaluation, benchmark accuracy scoring, and ablation studies.

        Computes average Character Error Rate (CER):
            CER = Levenshtein(orig, drift) / max(len(orig), 1)
        Range: [0.0, 1.0].
        """
        cer_scores = []
        for o, d in zip(orig_texts, drift_texts):
            if o == d:
                cer_scores.append(0.0)
                continue
            o_sub = o[:200]
            d_sub = d[:200]
            denom = max(len(o_sub), 1)
            dist = edit_distance(o_sub, d_sub)
            cer_scores.append(min(1.0, dist / denom))

        return float(np.mean(cer_scores)) if cer_scores else 0.0

    def _compute_pairwise_cosine_similarity(
        self, orig_texts: List[str], drift_texts: List[str]
    ) -> float:
        """
        [DEPRECATED — Reference-Free Migration]
        Computed sample-by-sample pairwise TF-IDF cosine similarity across paired documents
        (X_i vs X'_i). Removed because it assumes paired parallel text samples.
        """
        combined = orig_texts + drift_texts
        try:
            vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
            tfidf_mat = vectorizer.fit_transform(combined)
            n = len(orig_texts)
            orig_mat = tfidf_mat[:n]
            drift_mat = tfidf_mat[n:]

            pair_sims = []
            for i in range(n):
                v_orig = np.asarray(orig_mat[i].todense())
                v_drift = np.asarray(drift_mat[i].todense())
                norm_o = np.linalg.norm(v_orig)
                norm_d = np.linalg.norm(v_drift)
                if norm_o > 0 and norm_d > 0:
                    sim = float(cosine_similarity(v_orig, v_drift)[0][0])
                    pair_sims.append(sim)
                else:
                    pair_sims.append(1.0)
            return float(np.clip(np.mean(pair_sims), 0.0, 1.0)) if pair_sims else 1.0
        except Exception:
            return 1.0
