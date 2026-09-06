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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


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
    """

    def __init__(self):
        self.model: Optional[Pipeline] = None
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

    def train(self, texts: List[str], scores: List[float], version: str = "v1.0") -> Dict[str, Any]:
        """
        Trains TF-IDF + Logistic Regression on provided texts and star scores.
        """
        if not texts or not scores or len(texts) != len(scores):
            return {"status": "error", "message": "Invalid training data: empty or mismatched lengths"}

        y = [score_to_sentiment(s) for s in scores]

        # Ensure at least 2 classes exist for classification; if single class, pad minimal anchors
        classes_present = set(y)
        train_texts = list(texts)
        train_y = list(y)
        if len(classes_present) < 2:
            if "positive" not in classes_present:
                train_texts.append("Excellent, brilliant, magnificent masterpiece!")
                train_y.append("positive")
            if "negative" not in classes_present:
                train_texts.append("Terrible, awful, dreadful, boring waste of time.")
                train_y.append("negative")

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2500, stop_words='english', ngram_range=(1, 2))),
            ('clf', LogisticRegression(C=1.0, max_iter=250, solver='lbfgs', random_state=42))
        ])

        pipeline.fit(train_texts, train_y)
        preds = pipeline.predict(train_texts)
        acc = float(np.mean(preds == train_y))

        self.model = pipeline
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
        if not self.is_trained or self.model is None or not text.strip():
            return {
                "prediction": "neutral",
                "confidence": 0.5,
                "probabilities": {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
            }

        proba = self.model.predict_proba([text])[0]
        class_idx = int(np.argmax(proba))
        pred_class = str(self.model.classes_[class_idx])
        conf = float(proba[class_idx])
        probs_dict = {str(c): round(float(p), 3) for c, p in zip(self.model.classes_, proba)}

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
