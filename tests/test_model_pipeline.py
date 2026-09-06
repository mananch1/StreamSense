"""
Unit Tests for StreamSense Closed-Loop MLOps Model Pipeline.
Tests:
- score_to_sentiment label mapping
- SentimentModel training on warm-up baseline
- Streaming inference & item evaluation
- Concept drift degradation simulation
- On-demand model retraining from accumulated stream data
- Integration with FeedSimulator
"""

import unittest
from src.sentiment_model import SentimentModel, score_to_sentiment
from src.data_loader import DataLoader
from src.drift_engine import DriftEngine
from src.feed_simulator import FeedSimulator
from src.config import DriftConfig


class TestSentimentModelPipeline(unittest.TestCase):
    def setUp(self):
        self.model = SentimentModel()
        self.clean_texts = [
            "An absolute masterpiece of cinematic history with magnificent performances.",
            "Great acting and compelling narrative with wonderful cinematography.",
            "Stunning visuals, magnificent score, and an unforgettable climax.",
            "Solid direction and wonderfully atmospheric, highly recommended.",
            "Completely overrated, boring, dreadful dialogue, total waste of time.",
            "Fails to live up to expectations, terrible acting and sloppy editing.",
            "Horrible script, unlikable characters, awful and painful to watch.",
            "Decent watch but nothing revolutionary, average weekend movie."
        ]
        self.clean_scores = [5.0, 4.0, 5.0, 4.0, 1.0, 2.0, 1.0, 3.0]

    def test_score_to_sentiment_mapping(self):
        self.assertEqual(score_to_sentiment(5.0), "positive")
        self.assertEqual(score_to_sentiment(4.0), "positive")
        self.assertEqual(score_to_sentiment(3.0), "neutral")
        self.assertEqual(score_to_sentiment(2.0), "negative")
        self.assertEqual(score_to_sentiment(1.0), "negative")

    def test_baseline_training_and_prediction(self):
        res = self.model.train(self.clean_texts, self.clean_scores, version="v1.0")
        self.assertTrue(self.model.is_trained)
        self.assertEqual(self.model.version, "v1.0")
        self.assertGreaterEqual(res["train_accuracy"], 0.85)

        # Test inference on positive review
        pred_pos = self.model.predict("Magnificent acting and extraordinary direction, brilliant movie!")
        self.assertEqual(pred_pos["prediction"], "positive")
        self.assertGreater(pred_pos["confidence"], 0.45)

        # Test inference on negative review
        pred_neg = self.model.predict("Dreadful, horrible, waste of two hours. Terrible film.")
        self.assertEqual(pred_neg["prediction"], "negative")
        self.assertGreater(pred_neg["confidence"], 0.45)

    def test_item_evaluation_and_degradation(self):
        self.model.train(self.clean_texts, self.clean_scores, version="v1.0")

        # 1. Clean positive item: should match
        eval_clean = self.model.evaluate_item(
            text="Magnificent acting and brilliant direction!",
            score=5.0
        )
        self.assertEqual(eval_clean["model_prediction"], "positive")
        self.assertEqual(eval_clean["ground_truth_label"], "positive")
        self.assertTrue(eval_clean["prediction_correct"])

        # 2. Drifted item (Class Swap: 5★ becomes 1★ label inversion)
        eval_drifted = self.model.evaluate_item(
            text="Magnificent acting and brilliant direction!",
            score=1.0  # swapped label
        )
        self.assertEqual(eval_drifted["model_prediction"], "positive")
        self.assertEqual(eval_drifted["ground_truth_label"], "negative")
        self.assertFalse(eval_drifted["prediction_correct"])

    def test_flush_window_metrics_and_status(self):
        self.model.train(self.clean_texts, self.clean_scores, version="v1.0")

        # Simulate 4 correct and 1 wrong
        for _ in range(4):
            self.model.evaluate_item("Magnificent acting and brilliant direction!", 5.0)
        self.model.evaluate_item("Magnificent acting and brilliant direction!", 1.0)

        metrics = self.model.flush_window_metrics()
        self.assertEqual(metrics["model_version"], "v1.0")
        self.assertEqual(metrics["model_window_accuracy"], 0.8)
        self.assertEqual(metrics["model_status"], "healthy")

        # Simulate severe degradation in next window (all misclassified)
        for _ in range(5):
            self.model.evaluate_item("Magnificent acting and brilliant direction!", 1.0)

        degraded_metrics = self.model.flush_window_metrics()
        self.assertEqual(degraded_metrics["model_window_accuracy"], 0.0)
        self.assertEqual(degraded_metrics["model_status"], "degraded")

    def test_model_retraining(self):
        # Initial training
        self.model.train(self.clean_texts, self.clean_scores, version="v1.0")
        self.assertEqual(self.model.version, "v1.0")

        # Retrain with augmented/drifted corpus to v2.0
        retrain_texts = self.clean_texts + [
            "Totally unwatchable and boring garbage.",
            "Phenomenal cast and legendary score!"
        ]
        retrain_scores = self.clean_scores + [1.0, 5.0]

        res = self.model.train(retrain_texts, retrain_scores, version="v2.0")
        self.assertEqual(self.model.version, "v2.0")
        self.assertEqual(res["sample_count"], len(retrain_texts))
        self.assertGreaterEqual(res["train_accuracy"], 0.85)

    def test_feed_simulator_retrain_integration(self):
        loader = DataLoader(mode="sample")
        engine = DriftEngine()
        feed = FeedSimulator(data_loader=loader, drift_engine=engine)

        # Buffer some simulated items into accumulated_stream_items
        feed.accumulated_stream_items = [
            {"drifted_text": "Great movie with wonderful performances", "drifted_score": 5.0, "timestamp": 1.0},
            {"drifted_text": "Horrible, terrible waste of time", "drifted_score": 1.0, "timestamp": 2.0},
            {"drifted_text": "Average film, not bad", "drifted_score": 3.0, "timestamp": 3.0}
        ] * 5

        retrain_res = feed.retrain_model()
        self.assertEqual(retrain_res["status"], "trained")
        self.assertEqual(retrain_res["version"], "v2.0")
        self.assertTrue(feed.sentiment_model.is_trained)
        self.assertTrue(retrain_res.get("baseline_recalibrated"))
        self.assertTrue(feed.metrics_calculator.baseline_ready)

        # Check subsequent window computes drift against the NEW baseline
        test_window = [
            {"drifted_text": "Great movie with wonderful performances", "drifted_score": 5.0, "is_drifted": False},
            {"drifted_text": "Horrible, terrible waste of time", "drifted_score": 1.0, "is_drifted": False},
            {"drifted_text": "Average film, not bad", "drifted_score": 3.0, "is_drifted": False}
        ] * 2
        metrics = feed.metrics_calculator.compute_window_metrics(test_window)
        self.assertEqual(metrics["phase"], "monitoring")
        self.assertGreater(metrics["cosine_similarity"], 0.9)
        self.assertLess(metrics["drift_magnitude_pct"], 20.0)

        status = feed.get_status()
        self.assertIn("model_status", status)
        self.assertEqual(status["model_status"]["version"], "v2.0")


if __name__ == "__main__":
    unittest.main()
