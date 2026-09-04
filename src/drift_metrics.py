"""
StreamSense Drift Metrics.
=========================
Calculates statistical, semantic, and syntactic drift indicators between
original reference review streams and drifted streaming text windows.

Academic & Contemporary Industry Foundation:
--------------------------------------------
1. Garcia et al. (2024), "Methods for Generating Drift in Text Streams", arXiv:2403.12328
2. Hutto & Gilbert (2014), "VADER: A Parsimonious Rule-based Model for Sentiment Analysis", ICWSM
3. Evidently AI & Seldon Alibi-Detect: Contemporary streaming NLP drift detection standards
   (Jensen-Shannon Divergence, Earth Mover's / Wasserstein Distance, Vector Cosine Distance,
   and Character Error Rates)

Monitored Metrics:
------------------
1. Centroid & Pairwise TF-IDF Cosine Similarity [0.0, 1.0]
2. Vocabulary Jaccard Overlap [0.0, 1.0]
3. VADER Sentiment Polarity Distribution (Positive / Neutral / Negative)
4. Kullback-Leibler (KL) Divergence [0.0, +inf) with epsilon smoothing
5. Jensen-Shannon Divergence (JSD) [0.0, 1.0] (Symmetric, bounded relative entropy)
6. Sentiment Wasserstein-1 Distance (Earth Mover's Distance) [0.0, 2.0] for ordinal sentiment
7. Average Rating Score Delta (Mean Absolute Error on 1.0-5.0 star scale)
8. Character Mutation Rate / Character Error Rate (CER) via normalized edit distance [0.0, 1.0]
9. Composite Drift Magnitude Score [0.0%, 100.0%] (Multi-criteria weighted perturbation index)
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


class DriftMetricsCalculator:
    """
    Computes statistical and semantic data drift indicators across tumbling or sliding
    streaming windows of processed reviews.
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def compute_window_metrics(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes a window of processed review dicts and computes comprehensive drift metrics.

        Args:
            items: List of review dictionaries containing:
                   - 'original_text' (str): Baseline uncorrupted text
                   - 'drifted_text' (str): Streamed / perturbed text
                   - 'original_score' (float): Star rating before drift (1.0 to 5.0)
                   - 'drifted_score' (float): Star rating after potential class drift
                   - 'is_drifted' (bool): True if synthetic drift was applied

        Returns:
            Dict[str, Any] containing all statistical metrics and telemetry records.
        """
        if not items:
            return {
                "window_size": 0,
                "cosine_similarity": 1.0,
                "pairwise_cosine_similarity": 1.0,
                "vocab_overlap": 1.0,
                "sentiment_kl_divergence": 0.0,
                "sentiment_js_divergence": 0.0,
                "sentiment_wasserstein_distance": 0.0,
                "character_mutation_rate": 0.0,
                "original_sentiment_dist": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
                "drifted_sentiment_dist": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
                "drift_magnitude_pct": 0.0,
                "drifted_items_count": 0,
                "drifted_items_ratio": 0.0,
                "avg_score_delta": 0.0
            }

        orig_texts = [it.get("original_text", "") for it in items]
        drift_texts = [it.get("drifted_text", "") for it in items]
        orig_scores = [float(it.get("original_score", 3.0)) for it in items]
        drift_scores = [float(it.get("drifted_score", 3.0)) for it in items]

        # 1. TF-IDF Cosine Similarity (Centroid & Pairwise)
        # Fix: Convert scipy sparse mean (np.matrix) to np.asarray to prevent
        # TypeError in modern scikit-learn check_pairwise_arrays.
        centroid_cosine_sim, pairwise_cosine_sim = self._compute_cosine_similarity(orig_texts, drift_texts)

        # 2. Vocabulary Jaccard Overlap
        # Fix: Proper regex token extraction to avoid stripping tokens attached to punctuation.
        vocab_overlap = self._compute_vocab_overlap(orig_texts, drift_texts)

        # 3. Sentiment Distribution Shift (using VADER compound thresholds)
        orig_sent_dist = self._get_sentiment_distribution(orig_texts)
        drift_sent_dist = self._get_sentiment_distribution(drift_texts)

        # 4. Relative Entropy Divergences:
        # a) Classical Kullback-Leibler Divergence (asymmetric, with epsilon-smoothing)
        kl_div = self._compute_kl_divergence(orig_sent_dist, drift_sent_dist)
        # b) Jensen-Shannon Divergence (symmetric, bounded in [0, 1] using base-2 log)
        js_div = self._compute_js_divergence(orig_sent_dist, drift_sent_dist)
        # c) Wasserstein-1 Distance (Earth Mover's Distance) for ordinal sentiment (neg -> neu -> pos)
        wasserstein_dist = self._compute_sentiment_wasserstein(orig_sent_dist, drift_sent_dist)

        # 5. Character Mutation Rate / Character Error Rate (CER) via Levenshtein edit distance
        char_mutation_rate = self._compute_character_mutation_rate(orig_texts, drift_texts)

        # 6. Rating score deviation (Mean Absolute Error on 1-5 scale)
        score_deltas = [abs(d - o) for o, d in zip(orig_scores, drift_scores)]
        avg_score_delta = float(np.mean(score_deltas))

        # 7. Proportion of drifted items in window (empirical ground-truth rate)
        drifted_count = sum(1 for it in items if it.get("is_drifted", False))
        drift_prop = drifted_count / len(items)

        # 8. Composite Drift Magnitude Score (0% to 100%)
        # Combines semantic cosine distance, vocabulary drop, bounded divergence, and score delta.
        cosine_dist = max(0.0, 1.0 - centroid_cosine_sim)
        vocab_dist = max(0.0, 1.0 - vocab_overlap)
        # Use Jensen-Shannon Distance sqrt(JSD) in [0, 1] as the stable divergence component;
        # also cap normalized KL for backward compatibility:
        norm_divergence = math.sqrt(js_div)  # strictly bounded in [0.0, 1.0]
        norm_score_delta = min(1.0, avg_score_delta / 4.0)

        composite_drift = (
            0.35 * cosine_dist +
            0.25 * vocab_dist +
            0.20 * norm_divergence +
            0.20 * norm_score_delta
        ) * 100.0

        metrics_record = {
            "window_size": len(items),
            "cosine_similarity": round(float(centroid_cosine_sim), 4),
            "pairwise_cosine_similarity": round(float(pairwise_cosine_sim), 4),
            "vocab_overlap": round(float(vocab_overlap), 4),
            "sentiment_kl_divergence": round(float(kl_div), 4),
            "sentiment_js_divergence": round(float(js_div), 4),
            "sentiment_wasserstein_distance": round(float(wasserstein_dist), 4),
            "character_mutation_rate": round(float(char_mutation_rate), 4),
            "original_sentiment_dist": orig_sent_dist,
            "drifted_sentiment_dist": drift_sent_dist,
            "avg_score_delta": round(avg_score_delta, 2),
            "drift_magnitude_pct": round(float(composite_drift), 1),
            "drifted_items_count": drifted_count,
            "drifted_items_ratio": round(drift_prop, 3),
        }

        self.history.append(metrics_record)
        # Keep last 100 windows in memory
        if len(self.history) > 100:
            self.history.pop(0)

        return metrics_record

    def _compute_cosine_similarity(
        self, orig_texts: List[str], drift_texts: List[str]
    ) -> (float, float):
        """
        Computes TF-IDF vector space similarity between original and drifted texts.

        Returns two complementary measurements:
        1. Centroid Cosine Similarity: Cosine similarity between the mean document vectors
           of the original window vs drifted window.
        2. Pairwise Cosine Similarity: Average sample-by-sample cosine similarity across
           paired documents (X_i vs X'_i).
        """
        combined = orig_texts + drift_texts
        try:
            vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
            tfidf_mat = vectorizer.fit_transform(combined)
            n = len(orig_texts)
            orig_mat = tfidf_mat[:n]
            drift_mat = tfidf_mat[n:]

            # Scipy sparse matrix mean returns an np.matrix. In modern scikit-learn,
            # np.matrix raises a TypeError unless cast back to a standard np.ndarray.
            orig_centroid = np.asarray(orig_mat.mean(axis=0))
            drift_centroid = np.asarray(drift_mat.mean(axis=0))

            centroid_sim = float(cosine_similarity(orig_centroid, drift_centroid)[0][0])
            centroid_sim = float(np.clip(centroid_sim, 0.0, 1.0))

            # Sample-wise paired similarity
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
            pairwise_sim = float(np.clip(np.mean(pair_sims), 0.0, 1.0)) if pair_sims else centroid_sim

            return centroid_sim, pairwise_sim
        except Exception:
            return 1.0, 1.0

    def _compute_vocab_overlap(self, orig_texts: List[str], drift_texts: List[str]) -> float:
        """
        Computes vocabulary Jaccard similarity index J(A, B) = |A ∩ B| / |A ∪ B|.
        Uses regex tokenization to prevent dropping words with attached punctuation.
        """
        def get_words(texts: List[str]):
            words = set()
            for t in texts:
                # Extract alphabetic words of length >= 3 cleanly without punctuation
                tokens = re.findall(r'\b[a-zA-Z]{3,}\b', t.lower())
                words.update(tokens)
            return words

        orig_words = get_words(orig_texts)
        drift_words = get_words(drift_texts)

        if not orig_words or not drift_words:
            return 1.0

        intersection = len(orig_words.intersection(drift_words))
        union = len(orig_words.union(drift_words))
        return intersection / union if union > 0 else 1.0

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
        Note: KL divergence is asymmetric and unbounded [0, +inf).
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
        JSD is symmetric, finite, and its square root sqrt(JSD) is a true mathematical metric.
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
        Unlike KL or JSD (which treat classes as unordered nominal bins), Wasserstein
        naturally accounts for the semantic geometry of sentiment:
            Negative (0) < Neutral (1) < Positive (2)
        A shift from Positive to Negative incurs cost 2.0, whereas Positive to Neutral incurs 1.0.
        Range: [0.0, 2.0].
        """
        u_values = [0.0, 1.0, 2.0]  # Ordered: Negative, Neutral, Positive
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

    def _compute_character_mutation_rate(
        self, orig_texts: List[str], drift_texts: List[str]
    ) -> float:
        """
        Computes average Character Error Rate (CER) using normalized Levenshtein edit distance:
            CER = Levenshtein(orig, drift) / max(len(orig), 1)
        Quantifies syntactic noise, typographical mutations, and character-level corruptions.
        Range: [0.0, 1.0].
        """
        cer_scores = []
        for o, d in zip(orig_texts, drift_texts):
            if o == d:
                cer_scores.append(0.0)
                continue
            # Sample first 200 chars for real-time stream processing throughput
            o_sub = o[:200]
            d_sub = d[:200]
            denom = max(len(o_sub), 1)
            dist = edit_distance(o_sub, d_sub)
            cer_scores.append(min(1.0, dist / denom))

        return float(np.mean(cer_scores)) if cer_scores else 0.0
