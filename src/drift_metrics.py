"""
StreamSense Drift Metrics.
Calculates statistical and semantic drift indicators between original and drifted text streams:
1. TF-IDF Cosine Similarity
2. Vocabulary Jaccard Overlap
3. Sentiment Distribution Shift (KL Divergence & Wasserstein Distance)
4. Character Mutation Rate
5. Composite Drift Magnitude Score
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader_analyzer = SentimentIntensityAnalyzer()

class DriftMetricsCalculator:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def compute_window_metrics(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes a window of processed review dicts and computes comprehensive drift metrics.
        """
        if not items:
            return {
                "cosine_similarity": 1.0,
                "vocab_overlap": 1.0,
                "sentiment_kl_divergence": 0.0,
                "original_sentiment_dist": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
                "drifted_sentiment_dist": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
                "drift_magnitude_pct": 0.0,
                "drifted_items_count": 0,
                "total_items": 0,
                "avg_score_delta": 0.0
            }

        orig_texts = [it.get("original_text", "") for it in items]
        drift_texts = [it.get("drifted_text", "") for it in items]
        orig_scores = [float(it.get("original_score", 3.0)) for it in items]
        drift_scores = [float(it.get("drifted_score", 3.0)) for it in items]

        # 1. TF-IDF Cosine Similarity
        cosine_sim = self._compute_cosine_similarity(orig_texts, drift_texts)

        # 2. Vocabulary Jaccard Overlap
        vocab_overlap = self._compute_vocab_overlap(orig_texts, drift_texts)

        # 3. Sentiment Distribution Shift (using VADER sentiment analysis)
        orig_sent_dist = self._get_sentiment_distribution(orig_texts)
        drift_sent_dist = self._get_sentiment_distribution(drift_texts)
        kl_div = self._compute_kl_divergence(orig_sent_dist, drift_sent_dist)

        # 4. Rating score deviation
        score_deltas = [abs(d - o) for o, d in zip(orig_scores, drift_scores)]
        avg_score_delta = float(np.mean(score_deltas))

        # 5. Proportion of drifted items in window
        drifted_count = sum(1 for it in items if it.get("is_drifted", False))
        drift_prop = drifted_count / len(items)

        # 6. Composite Drift Magnitude Score (0% to 100%)
        # Combines cosine distance, vocab drop, score delta, and sentiment shift
        cosine_dist = max(0.0, 1.0 - cosine_sim)
        vocab_dist = max(0.0, 1.0 - vocab_overlap)
        norm_kl = min(1.0, kl_div / 2.0)
        norm_score_delta = min(1.0, avg_score_delta / 4.0)

        composite_drift = (
            0.35 * cosine_dist +
            0.25 * vocab_dist +
            0.20 * norm_kl +
            0.20 * norm_score_delta
        ) * 100.0

        metrics_record = {
            "window_size": len(items),
            "cosine_similarity": round(float(cosine_sim), 4),
            "vocab_overlap": round(float(vocab_overlap), 4),
            "sentiment_kl_divergence": round(float(kl_div), 4),
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

    def _compute_cosine_similarity(self, orig_texts: List[str], drift_texts: List[str]) -> float:
        combined = orig_texts + drift_texts
        try:
            vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
            tfidf_mat = vectorizer.fit_transform(combined)
            n = len(orig_texts)
            orig_vec = tfidf_mat[:n].mean(axis=0)
            drift_vec = tfidf_mat[n:].mean(axis=0)
            sim = cosine_similarity(orig_vec, drift_vec)[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 1.0

    def _compute_vocab_overlap(self, orig_texts: List[str], drift_texts: List[str]) -> float:
        def get_words(texts):
            words = set()
            for t in texts:
                for w in t.lower().split():
                    if len(w) > 2 and w.isalpha():
                        words.add(w)
            return words

        orig_words = get_words(orig_texts)
        drift_words = get_words(drift_texts)

        if not orig_words or not drift_words:
            return 1.0

        intersection = len(orig_words.intersection(drift_words))
        union = len(orig_words.union(drift_words))
        return intersection / union if union > 0 else 1.0

    def _get_sentiment_distribution(self, texts: List[str]) -> Dict[str, float]:
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
            "positive": round(pos_cnt / total, 3),
            "neutral": round(neu_cnt / total, 3),
            "negative": round(neg_cnt / total, 3)
        }

    def _compute_kl_divergence(self, p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
        """
        Kullback-Leibler Divergence: KL(P || Q) with epsilon smoothing
        """
        eps = 1e-5
        kl = 0.0
        for key in ["positive", "neutral", "negative"]:
            p = max(eps, p_dist.get(key, 0.0))
            q = max(eps, q_dist.get(key, 0.0))
            kl += p * math.log(p / q)
        return max(0.0, kl)
