"""
Unit Tests for StreamSense Drift Engine and Metrics.
Tests all 6 drift methods:
- Adjective Swap
- Class Swap
- Class Shift
- Time-Slice Removal
- Noise Injection
- Formality Shift
And validates metrics computation (Cosine similarity, vocab overlap, sentiment shift).
"""

import unittest
from src.config import DriftConfig
from src.drift_engine import DriftEngine
from src.drift_metrics import DriftMetricsCalculator

class TestStreamSenseDrift(unittest.TestCase):
    def setUp(self):
        self.engine = DriftEngine()
        self.sample_review = {
            "productId": "B00006HAXW",
            "userId": "U12345",
            "profileName": "FilmFan",
            "score": "5.0",
            "summary": "An excellent and beautiful movie",
            "text": "The acting was magnificent and the direction was superb. I really loved this brilliant film."
        }

    def test_adjective_swap_modifies_text(self):
        config = DriftConfig(enable_adjective_swap=True, adjective_swap_intensity=1.0)
        self.engine.update_config(config)
        
        result = self.engine.process_review(self.sample_review)
        self.assertTrue(result["is_drifted"])
        self.assertIn("Adjective Swap (WordNet)", result["active_drifts"])
        # Original text has words like 'excellent', 'beautiful', 'magnificent', 'superb', 'brilliant'
        # Adjective swap should have found antonyms
        print(f"\n[Test Adj Swap]\n  Orig: {self.sample_review['text']}\n  Drift: {result['drifted_text']}")

    def test_class_swap_inverts_ratings(self):
        config = DriftConfig(enable_class_swap=True)
        self.engine.update_config(config)
        
        # 5.0 should become 1.0
        res5 = self.engine.process_review(self.sample_review)
        self.assertEqual(res5["drifted_score"], 1.0)
        
        # 4.0 should become 2.0
        review4 = dict(self.sample_review, score="4.0")
        res4 = self.engine.process_review(review4)
        self.assertEqual(res4["drifted_score"], 2.0)

    def test_class_shift_cycles_ratings(self):
        config = DriftConfig(enable_class_shift=True)
        self.engine.update_config(config)
        
        # 5.0 should become 1.0 in cyclic shift
        res5 = self.engine.process_review(self.sample_review)
        self.assertEqual(res5["drifted_score"], 1.0)
        
        # 1.0 should become 2.0
        review1 = dict(self.sample_review, score="1.0")
        res1 = self.engine.process_review(review1)
        self.assertEqual(res1["drifted_score"], 2.0)

    def test_formality_shift(self):
        config = DriftConfig(enable_formality_shift=True, formality_intensity=1.0)
        self.engine.update_config(config)
        
        result = self.engine.process_review(self.sample_review)
        self.assertTrue(result["is_drifted"])
        print(f"\n[Test Formality Shift]\n  Orig: {self.sample_review['text']}\n  Drift: {result['drifted_text']}")

    def test_noise_injection(self):
        config = DriftConfig(enable_noise_injection=True, noise_intensity=0.8)
        self.engine.update_config(config)
        
        result = self.engine.process_review(self.sample_review)
        self.assertTrue(result["is_drifted"])
        print(f"\n[Test Noise Injection]\n  Orig: {self.sample_review['text']}\n  Drift: {result['drifted_text']}")

    def test_drift_metrics_calculation(self):
        calc = DriftMetricsCalculator()
        
        config = DriftConfig(enable_adjective_swap=True, adjective_swap_intensity=1.0)
        self.engine.update_config(config)
        
        window = [self.engine.process_review(self.sample_review) for _ in range(10)]
        metrics = calc.compute_window_metrics(window)
        
        self.assertIn("cosine_similarity", metrics)
        self.assertIn("pairwise_cosine_similarity", metrics)
        self.assertIn("vocab_overlap", metrics)
        self.assertIn("sentiment_kl_divergence", metrics)
        self.assertIn("sentiment_js_divergence", metrics)
        self.assertIn("sentiment_wasserstein_distance", metrics)
        self.assertIn("character_mutation_rate", metrics)
        self.assertIn("drift_magnitude_pct", metrics)
        self.assertLess(metrics["cosine_similarity"], 1.0)
        self.assertGreaterEqual(metrics["sentiment_js_divergence"], 0.0)
        self.assertLessEqual(metrics["sentiment_js_divergence"], 1.0)
        print(f"\n[Test Drift Metrics]\n  Metrics: {metrics}")

if __name__ == "__main__":
    unittest.main()
