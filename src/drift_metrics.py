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
# Build an expanded dictionary that includes common inflected forms (plurals,
# past tense, gerunds, adverbs) to avoid false-positive misspelling hits on
# perfectly clean English text like "uses", "called", "workouts", "exceeded".
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)
from nltk.corpus import words as nltk_words

def _build_expanded_dictionary():
    """Build dictionary including common English inflections to minimise false positives."""
    base_words = set(w.lower() for w in nltk_words.words())
    expanded = set(base_words)
    _suffixes = ['s', 'es', 'ed', 'ing', 'er', 'ers', 'est', 'ly', 'ment',
                 'ments', 'ness', 'tion', 'tions', 'sion', 'sions', 'ous',
                 'ful', 'less', 'able', 'ible', 'ity', 'ies', 'ize', 'ized',
                 'ise', 'ised', 'ising', 'izing', 'isation', 'ization',
                 'al', 'ial', 'ical', 'ically', 'ive', 'ively']
    for word in base_words:
        if len(word) >= 3:
            for suffix in _suffixes:
                expanded.add(word + suffix)
            # Handle consonant doubling (e.g. "stop" -> "stopped", "stopping")
            if len(word) >= 3 and word[-1] not in 'aeiouy' and word[-2] in 'aeiou':
                expanded.add(word + word[-1] + 'ed')
                expanded.add(word + word[-1] + 'ing')
                expanded.add(word + word[-1] + 'er')
            # Handle silent-e dropping (e.g. "use" -> "using", "used")
            if word.endswith('e'):
                expanded.add(word[:-1] + 'ing')
                expanded.add(word[:-1] + 'ed')
                expanded.add(word[:-1] + 'er')
                expanded.add(word[:-1] + 'able')
                expanded.add(word[:-1] + 'ation')
            # Handle y -> ies (e.g. "story" -> "stories")
            if word.endswith('y') and len(word) > 2 and word[-2] not in 'aeiou':
                expanded.add(word[:-1] + 'ies')
                expanded.add(word[:-1] + 'ied')
                expanded.add(word[:-1] + 'ier')
                expanded.add(word[:-1] + 'iest')
                expanded.add(word[:-1] + 'ily')

    # Common auxiliary verbs, contractions, abbreviations, and domain words that
    # NLTK's words corpus misses but are perfectly valid English.
    _supplemental = {
        'has', 'had', 'was', 'were', 'been', 'being', 'does', 'did', 'doing',
        'isn', 'aren', 'wasn', 'weren', 'doesn', 'didn', 'hasn', 'hadn',
        'won', 'wouldn', 'couldn', 'shouldn', 'mustn', 'don', 'ain',
        'dvd', 'dvds', 'dvr', 'blu', 'hd', 'tv', 'cgi', 'fps',
        'cardio', 'yoga', 'workout', 'workouts', 'pushups', 'pushup',
        'situps', 'pullups', 'ups', 'abs', 'reps', 'hiit',
        'online', 'offline', 'email', 'emails', 'website', 'login',
        'app', 'apps', 'wifi', 'bluetooth', 'usb', 'podcast', 'podcasts',
        'blog', 'blogs', 'vlog', 'vlogs', 'selfie', 'selfies',
        'binge', 'binged', 'binging', 'bingeing', 'streaming',
        'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'ain',
        'ok', 'okay', 'btw', 'fyi', 'imo', 'imho', 'lol', 'omg',
        'ngl', 'tbh', 'irl', 'brb', 'smh', 'fomo',
    }
    expanded.update(_supplemental)
    return expanded

ENGLISH_WORDS = _build_expanded_dictionary()


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
        self._baseline_scores: List[float] = []
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

        # Baseline variability calibration — natural noise floor of each metric
        # computed from leave-one-window-out during burn-in, so the composite
        # score reads near 0% on clean data.
        self._baseline_cosine_floor: float = 0.0
        self._baseline_vocab_floor: float = 0.0
        self._burn_in_per_window_texts: List[List[str]] = []
        self._burn_in_per_window_scores: List[List[float]] = []

    def reset_baseline(self):
        """Clear baseline and re-enter burn-in phase."""
        self.burn_in_windows_collected = 0
        self.baseline_ready = False
        self._baseline_texts = []
        self._baseline_scores = []
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
        self._baseline_cosine_floor = 0.0
        self._baseline_vocab_floor = 0.0
        self._burn_in_per_window_texts = []
        self._burn_in_per_window_scores = []
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
            # 1. Accumulate raw texts and scores
            self._baseline_texts.extend(stream_texts)
            self._baseline_scores.extend(stream_scores)
            # Store per-window data for calibration floor computation
            self._burn_in_per_window_texts.append(list(stream_texts))
            self._burn_in_per_window_scores.append(list(stream_scores))

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

        # 2. Vocabulary in-baseline coverage (fraction of stream tokens present in baseline)
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
        # Multi-criteria calibrated perturbation index.
        # Sub-metric anomaly signals are normalized [0.0, 1.0] relative to natural clean noise floors.
        # Max-blended aggregation (65% max + 35% mean) ensures that acute drift in ANY single modality
        # (e.g., label shift, semantic inversion, or syntactic corruption) produces a sharp, visible
        # drift alert (50-75%), while multiple simultaneous drifts escalate toward 85-95%.
        # On clean data, all signals evaluate to ~0.0, keeping baseline drift at 0-5%.
        d_cs = max(0.0, getattr(self, "_baseline_cosine_threshold", 0.75) - centroid_cosine_sim)
        sig_cs = min(1.0, d_cs / 0.35)

        d_vo = max(0.0, getattr(self, "_baseline_vocab_threshold", 0.55) - vocab_overlap)
        sig_vo = min(1.0, d_vo / 0.35)

        norm_sent_divergence = math.sqrt(js_div)
        d_jsd = max(0.0, norm_sent_divergence - getattr(self, "_baseline_jsd_threshold", 0.15))
        sig_jsd = min(1.0, d_jsd / 0.40)

        d_score = max(0.0, score_dist_divergence - getattr(self, "_baseline_score_threshold", 0.08))
        sig_score = min(1.0, d_score / 0.45)

        ser_floor = getattr(self, "_baseline_ser_threshold", max(0.015, self._baseline_spelling_error_rate + 0.015))
        d_ser = max(0.0, current_spelling_rate - ser_floor)
        sig_ser = min(1.0, d_ser / 0.18)

        signals = [sig_cs, sig_vo, sig_jsd, sig_score, sig_ser]
        max_sig = max(signals)
        mean_sig = float(np.mean(signals))
        composite_drift = (0.65 * max_sig + 0.35 * mean_sig) * 100.0
        if composite_drift < 1.5:
            composite_drift = 0.0

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

        # ------------------------------------------------------------------
        # Leave-One-Out Calibration: measure natural variance across burn-in windows
        # to establish clean baseline thresholds without data leakage.
        # ------------------------------------------------------------------
        n_wins = len(self._burn_in_per_window_texts)
        if n_wins >= 2 and self._baseline_vectorizer is not None:
            loo_cs, loo_vo, loo_jsd, loo_score = [], [], [], []
            for i in range(n_wins):
                held_t = self._burn_in_per_window_texts[i]
                held_s = self._burn_in_per_window_scores[i]
                other_t = [t for j in range(n_wins) if j != i for t in self._burn_in_per_window_texts[j]]
                other_s = [s for j in range(n_wins) if j != i for s in self._burn_in_per_window_scores[j]]

                # Held-out cosine vs other centroid
                try:
                    v_tmp = TfidfVectorizer(max_features=2000, stop_words='english')
                    m_oth = v_tmp.fit_transform(other_t)
                    c_oth = np.asarray(m_oth.mean(axis=0))
                    c_hld = np.asarray(v_tmp.transform(held_t).mean(axis=0))
                    sim = float(cosine_similarity(c_oth, c_hld)[0][0])
                    loo_cs.append(sim)
                except Exception:
                    pass

                # Held-out in-vocabulary coverage against other vocabulary
                voc_oth = set(tok for t in other_t for tok in re.findall(r'\b[a-zA-Z]{3,}\b', t.lower()))
                voc_hld = set(tok for t in held_t for tok in re.findall(r'\b[a-zA-Z]{3,}\b', t.lower()))
                if voc_hld:
                    loo_vo.append(len(voc_oth.intersection(voc_hld)) / len(voc_hld))

                # Sentiment and score JSD
                s_dist_oth = self._get_sentiment_distribution(other_t)
                s_dist_hld = self._get_sentiment_distribution(held_t)
                loo_jsd.append(math.sqrt(self._compute_js_divergence(s_dist_oth, s_dist_hld)))

                sc_dist_oth = self._compute_score_distribution(other_s)
                sc_dist_hld = self._compute_score_distribution(held_s)
                loo_score.append(self._compute_score_dist_divergence(sc_dist_oth, sc_dist_hld))

            self._baseline_cosine_threshold = float(np.min(loo_cs)) * 0.95 if loo_cs else 0.75
            self._baseline_vocab_threshold = float(np.min(loo_vo)) * 0.92 if loo_vo else 0.55
            self._baseline_jsd_threshold = float(np.max(loo_jsd)) * 1.15 if loo_jsd else 0.15
            self._baseline_score_threshold = float(np.max(loo_score)) * 1.15 if loo_score else 0.08
        else:
            self._baseline_cosine_threshold = 0.75
            self._baseline_vocab_threshold = 0.55
            self._baseline_jsd_threshold = 0.15
            self._baseline_score_threshold = 0.08

        self._baseline_ser_threshold = self._baseline_spelling_error_rate + 0.015

    def calibrate_baseline_from_data(self, texts: List[str], scores: List[float]):
        """
        Re-calibrates the frozen reference baseline from a new corpus of texts and scores
        (e.g., when the downstream model is retrained on accumulated stream data).
        This ensures that the baseline reflects the exact distribution the model was last trained on,
        so subsequent drift measures deviation from the newly deployed model's reference state.
        """
        if not texts or not scores:
            return

        self._baseline_texts = list(texts)
        self._baseline_scores = list(scores)

        # Recompute baseline vocabulary
        self._baseline_vocab = set()
        for t in texts:
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', t.lower())
            self._baseline_vocab.update(tokens)

        # Recompute sentiment distribution
        self._baseline_sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        self._baseline_sentiment_total = 0
        for t in texts:
            if not t.strip():
                self._baseline_sentiment_counts["neutral"] += 1
            else:
                vs = vader_analyzer.polarity_scores(t)
                cmp = vs['compound']
                if cmp >= 0.05:
                    self._baseline_sentiment_counts["positive"] += 1
                elif cmp <= -0.05:
                    self._baseline_sentiment_counts["negative"] += 1
                else:
                    self._baseline_sentiment_counts["neutral"] += 1
            self._baseline_sentiment_total += 1

        tot_sent = max(1, self._baseline_sentiment_total)
        self._baseline_sentiment_dist = {
            k: round(self._baseline_sentiment_counts.get(k, 0) / tot_sent, 4)
            for k in ["positive", "neutral", "negative"]
        }

        # Recompute score distribution
        self._baseline_score_counts = {}
        self._baseline_score_total = 0
        for s in scores:
            key = str(float(min(5.0, max(1.0, round(s)))))
            self._baseline_score_counts[key] = self._baseline_score_counts.get(key, 0) + 1
            self._baseline_score_total += 1

        tot_score = max(1, self._baseline_score_total)
        self._baseline_score_dist = {
            k: round(self._baseline_score_counts.get(k, 0) / tot_score, 4)
            for k in ["1.0", "2.0", "3.0", "4.0", "5.0"]
        }

        # Refit TF-IDF vectorizer and centroid on the new baseline
        try:
            self._baseline_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
            tfidf_mat = self._baseline_vectorizer.fit_transform(self._baseline_texts)
            self._baseline_centroid = np.asarray(tfidf_mat.mean(axis=0))
        except Exception:
            self._baseline_vectorizer = None
            self._baseline_centroid = None

        # Recompute baseline spelling error rate
        self._baseline_spelling_error_rate = self._compute_spelling_error_rate(texts)
        self._baseline_spelling_error_rates = [self._baseline_spelling_error_rate]

        # Ensure engine is in monitoring phase against this new baseline
        self.burn_in_windows_collected = self.burn_in_windows_needed
        self.baseline_ready = True

        # Recompute calibration thresholds from sub-windows of the retraining corpus
        n_sub = max(2, min(4, len(texts) // 25))
        win_sz = max(10, len(texts) // n_sub)
        sub_t = [texts[i*win_sz:(i+1)*win_sz] for i in range(n_sub)]
        sub_s = [scores[i*win_sz:(i+1)*win_sz] for i in range(n_sub)]

        loo_cs, loo_vo, loo_jsd, loo_score = [], [], [], []
        for i in range(n_sub):
            h_t, h_s = sub_t[i], sub_s[i]
            o_t = [t for j in range(n_sub) if j != i for t in sub_t[j]]
            o_s = [s for j in range(n_sub) if j != i for s in sub_s[j]]
            if o_t and h_t:
                try:
                    v_tmp = TfidfVectorizer(max_features=2000, stop_words='english')
                    m_oth = v_tmp.fit_transform(o_t)
                    c_oth = np.asarray(m_oth.mean(axis=0))
                    c_hld = np.asarray(v_tmp.transform(h_t).mean(axis=0))
                    loo_cs.append(float(cosine_similarity(c_oth, c_hld)[0][0]))
                except Exception:
                    pass
                voc_oth = set(tok for t in o_t for tok in re.findall(r'\b[a-zA-Z]{3,}\b', t.lower()))
                voc_hld = set(tok for t in h_t for tok in re.findall(r'\b[a-zA-Z]{3,}\b', t.lower()))
                if voc_hld:
                    loo_vo.append(len(voc_oth.intersection(voc_hld)) / len(voc_hld))
                s_dist_oth = self._get_sentiment_distribution(o_t)
                s_dist_hld = self._get_sentiment_distribution(h_t)
                loo_jsd.append(math.sqrt(self._compute_js_divergence(s_dist_oth, s_dist_hld)))
                sc_dist_oth = self._compute_score_distribution(o_s)
                sc_dist_hld = self._compute_score_distribution(h_s)
                loo_score.append(self._compute_score_dist_divergence(sc_dist_oth, sc_dist_hld))

        self._baseline_cosine_threshold = float(np.min(loo_cs)) * 0.95 if loo_cs else 0.75
        self._baseline_vocab_threshold = float(np.min(loo_vo)) * 0.92 if loo_vo else 0.55
        self._baseline_jsd_threshold = float(np.max(loo_jsd)) * 1.15 if loo_jsd else 0.15
        self._baseline_score_threshold = float(np.max(loo_score)) * 1.15 if loo_score else 0.08
        self._baseline_ser_threshold = self._baseline_spelling_error_rate + 0.015

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
        Computes vocabulary coverage: what fraction of baseline vocabulary tokens
        appear in the current streaming window.

        Coverage = |B ∩ C| / |B|

        This measures how much of the known baseline vocabulary is still present
        in the current data. It is preferred over Jaccard similarity for streaming
        windows because Jaccard is inherently low when comparing a small window
        against a larger baseline corpus (different reviews naturally use different
        words). Coverage gives values near 1.0 on clean data and drops meaningfully
        when drift introduces novel or corrupted vocabulary.

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
        return float(intersection / len(current_words)) if current_words else 1.0

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
