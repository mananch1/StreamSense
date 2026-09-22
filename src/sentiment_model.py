"""
StreamSense Sentiment Model Pipeline.
====================================
Lightweight streaming sentiment classifier demonstrating closed-loop MLOps:
1. Warm-up baseline training (Model v1.0)
2. Streaming inference and accuracy observability
3. Degradation detection under synthetic concept drift
4. On-demand model retraining (Model v2.0+) using accumulated stream data
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def score_to_sentiment(score: float) -> str:
    """
    Maps 1.0 - 5.0 star review scores to 3 categorical sentiment labels:
      - score >= 4.0: 'positive' (4★, 5★)
      - score == 3.0: 'neutral'  (3★)
      - score <= 2.0: 'negative' (1★, 2★)
    """
    rounded = round(float(score))
    if rounded >= 4:
        return "positive"
    elif rounded <= 2:
        return "negative"
    else:
        return "neutral"


class SentimentModel:
    """
    Manages the lifecycle of the active streaming sentiment analysis model.
    Uses a hybrid TF-IDF + VADER sentiment feature representation with balanced
    class weighting so the model is sensitive to semantic, lexical, and syntactic text drift.
    """

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.model: Optional[Any] = None  # maintained for backward compatibility
        self.vader = SentimentIntensityAnalyzer()

        self.version: str = "untrained"
        self.trained_at_timestamp: Optional[str] = None
        self.training_sample_count: int = 0
        self.train_accuracy: float = 0.0
        self.is_trained: bool = False

        # Live performance observability counters
        self.total_inferences: int = 0
        self.total_correct: int = 0
        self.current_window_inferences: int = 0
        self.current_window_correct: int = 0
        self.last_window_accuracy: float = 1.0

    def _get_vader_features(self, texts: List[str]) -> np.ndarray:
        feats = []
        for t in texts:
            if not t or not t.strip():
                feats.append([0.0, 1.0, 0.0, 0.0])
            else:
                s = self.vader.polarity_scores(t)
                feats.append([s['pos'], s['neu'], s['neg'], s['compound']])
        return np.array(feats, dtype=np.float32)

    def _extract_features(self, texts: List[str], fit: bool = False):
        if fit:
            self.vectorizer = TfidfVectorizer(max_features=2500, stop_words='english', ngram_range=(1, 2))
            X_tfidf = self.vectorizer.fit_transform(texts)
        else:
            if self.vectorizer is None:
                raise ValueError("Model vectorizer has not been fitted.")
            X_tfidf = self.vectorizer.transform(texts)

        X_vader = self._get_vader_features(texts)
        # Scale VADER sentiment features (pos, neu, neg, compound) so the classifier
        # directly incorporates sentiment polarity priors that respond immediately to
        # semantic inversions (antonyms) and syntactic degradation (noise / typos).
        return hstack([X_tfidf, X_vader * 3.5])

    def train(self, texts: List[str], scores: List[float], version: str = "v1.0") -> Dict[str, Any]:
        """
        Trains Hybrid TF-IDF + VADER Logistic Regression on provided texts and star scores.
        """
        if not texts or not scores or len(texts) != len(scores):
            return {"status": "error", "message": "Invalid training data: empty or mismatched lengths"}

        y = [score_to_sentiment(s) for s in scores]

        # Ensure representation across all 3 classes so multi-class classification is stable
        train_texts = list(texts)
        train_y = list(y)
        classes_present = set(y)
        if "positive" not in classes_present:
            train_texts.append("Excellent, brilliant, magnificent masterpiece!")
            train_y.append("positive")
        if "neutral" not in classes_present:
            train_texts.append("Decent, average, ordinary watch, nothing special.")
            train_y.append("neutral")
        if "negative" not in classes_present:
            train_texts.append("Terrible, awful, dreadful, boring waste of time.")
            train_y.append("negative")

        X = self._extract_features(train_texts, fit=True)
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=300,
            solver='lbfgs',
            class_weight='balanced',
            random_state=42
        )
        self.classifier.fit(X, train_y)
        self.model = self.classifier

        preds = self.classifier.predict(X)
        acc = float(np.mean(preds == train_y))

        self.version = version
        self.is_trained = True
        self.training_sample_count = len(texts)
        self.train_accuracy = round(acc, 4)
        self.trained_at_timestamp = time.strftime("%H:%M:%S")

        # Reset cumulative tracking for the new model version
        self.total_inferences = 0
        self.total_correct = 0
        self.current_window_inferences = 0
        self.current_window_correct = 0
        self.last_window_accuracy = 1.0

        return {
            "status": "trained",
            "version": self.version,
            "sample_count": len(texts),
            "train_accuracy": self.train_accuracy,
            "trained_at": self.trained_at_timestamp
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Runs inference on a single text.
        """
        if not self.is_trained or self.classifier is None or not text.strip():
            return {
                "prediction": "neutral",
                "confidence": 0.5,
                "probabilities": {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
            }

        X = self._extract_features([text], fit=False)
        proba = self.classifier.predict_proba(X)[0]
        class_idx = int(np.argmax(proba))
        pred_class = str(self.classifier.classes_[class_idx])
        conf = float(proba[class_idx])
        probs_dict = {str(c): round(float(p), 3) for c, p in zip(self.classifier.classes_, proba)}

        return {
            "prediction": pred_class,
            "confidence": round(conf, 3),
            "probabilities": probs_dict
        }

    def evaluate_item(self, text: str, score: float) -> Dict[str, Any]:
        """
        Evaluates a streamed item: predicts sentiment, compares against rating label,
        and updates live observability metrics.
        """
        actual_label = score_to_sentiment(score)

        if not self.is_trained:
            return {
                "model_version": self.version,
                "model_prediction": "calibrating",
                "model_confidence": 0.0,
                "ground_truth_label": actual_label,
                "prediction_correct": True  # Don't penalize during burn-in
            }

        pred_info = self.predict(text)
        pred_class = pred_info["prediction"]
        is_correct = (pred_class == actual_label)

        self.total_inferences += 1
        if is_correct:
            self.total_correct += 1

        self.current_window_inferences += 1
        if is_correct:
            self.current_window_correct += 1

        return {
            "model_version": self.version,
            "model_prediction": pred_class,
            "model_confidence": pred_info["confidence"],
            "ground_truth_label": actual_label,
            "prediction_correct": is_correct
        }

    def flush_window_metrics(self) -> Dict[str, Any]:
        """
        Computes window accuracy, cumulative accuracy, and health status,
        then resets window-specific counters.
        """
        if not self.is_trained:
            return {
                "model_version": self.version,
                "model_window_accuracy": 1.0,
                "model_cumulative_accuracy": 1.0,
                "model_status": "calibrating",
                "model_inferences_total": 0,
                "training_sample_count": 0,
                "train_accuracy": 0.0
            }

        win_acc = (
            float(self.current_window_correct / self.current_window_inferences)
            if self.current_window_inferences > 0
            else self.last_window_accuracy
        )
        self.last_window_accuracy = win_acc

        cum_acc = (
            float(self.total_correct / self.total_inferences)
            if self.total_inferences > 0
            else win_acc
        )

        # Health status thresholds
        if win_acc >= 0.75:
            status = "healthy"
        elif win_acc >= 0.55:
            status = "at_risk"
        else:
            status = "degraded"

        record = {
            "model_version": self.version,
            "model_window_accuracy": round(win_acc, 4),
            "model_cumulative_accuracy": round(cum_acc, 4),
            "model_status": status,
            "model_inferences_total": self.total_inferences,
            "training_sample_count": self.training_sample_count,
            "train_accuracy": self.train_accuracy,
            "trained_at": self.trained_at_timestamp
        }

        # Reset window counters
        self.current_window_inferences = 0
        self.current_window_correct = 0

        return record

    def get_status(self) -> Dict[str, Any]:
        """Returns model state for API queries."""
        cum_acc = (
            float(self.total_correct / self.total_inferences)
            if self.total_inferences > 0
            else 1.0
        )
        return {
            "is_trained": self.is_trained,
            "version": self.version,
            "trained_at": self.trained_at_timestamp,
            "training_sample_count": self.training_sample_count,
            "train_accuracy": self.train_accuracy,
            "total_inferences": self.total_inferences,
            "cumulative_accuracy": round(cum_acc, 4),
            "last_window_accuracy": round(self.last_window_accuracy, 4),
            "status": "healthy" if self.last_window_accuracy >= 0.75 else ("degraded" if self.last_window_accuracy < 0.55 else "at_risk") if self.is_trained else "calibrating"
        }

    def reset(self):
        """Resets model state back to untrained."""
        self.vectorizer = None
        self.classifier = None
        self.model = None
        self.version = "untrained"
        self.trained_at_timestamp = None
        self.training_sample_count = 0
        self.train_accuracy = 0.0
        self.is_trained = False
        self.total_inferences = 0
        self.total_correct = 0
        self.current_window_inferences = 0
        self.current_window_correct = 0
        self.last_window_accuracy = 1.0
