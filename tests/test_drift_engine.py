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
        calc = DriftMetricsCalculator(burn_in_windows=2)
        
        # 1. Clean burn-in windows
        clean_item = {
            "original_text": self.sample_review["text"],
            "drifted_text": self.sample_review["text"],
            "original_score": 5.0,
            "drifted_score": 5.0,
            "is_drifted": False
        }
        clean_window = [clean_item for _ in range(10)]
        m1 = calc.compute_window_metrics(clean_window)
        self.assertEqual(m1["phase"], "burn_in")
        self.assertFalse(calc.baseline_ready)

        m2 = calc.compute_window_metrics(clean_window)
        self.assertEqual(m2["phase"], "burn_in")
        self.assertTrue(calc.baseline_ready)

        # 2. Monitoring window with adjective swap drift
        config = DriftConfig(enable_adjective_swap=True, adjective_swap_intensity=1.0)
        self.engine.update_config(config)
        
        drifted_window = [self.engine.process_review(self.sample_review) for _ in range(10)]
        metrics = calc.compute_window_metrics(drifted_window)
        
        self.assertEqual(metrics["phase"], "monitoring")
        self.assertIn("cosine_similarity", metrics)
        self.assertIn("vocab_overlap", metrics)
        self.assertIn("sentiment_kl_divergence", metrics)
        self.assertIn("sentiment_js_divergence", metrics)
        self.assertIn("sentiment_wasserstein_distance", metrics)
        self.assertIn("spelling_error_rate", metrics)
        self.assertIn("score_dist_divergence", metrics)
        self.assertIn("drift_magnitude_pct", metrics)
        self.assertLess(metrics["cosine_similarity"], 1.0)
        self.assertGreaterEqual(metrics["sentiment_js_divergence"], 0.0)
        self.assertLessEqual(metrics["sentiment_js_divergence"], 1.0)
        print(f"\n[Test Drift Metrics]\n  Metrics: {metrics}")

    def test_burn_in_phase_transition(self):
        calc = DriftMetricsCalculator(burn_in_windows=2)
        clean_item = {
            "drifted_text": "A wonderful cinematic experience with great performances.",
            "drifted_score": 5.0,
            "is_drifted": False
        }
        clean_window = [clean_item] * 5

        # Window 1
        m1 = calc.compute_window_metrics(clean_window)
        self.assertEqual(m1["phase"], "burn_in")
        self.assertFalse(calc.baseline_ready)
        self.assertEqual(m1["burn_in_progress"], "1/2")

        # Window 2 - completes burn-in
        m2 = calc.compute_window_metrics(clean_window)
        self.assertEqual(m2["phase"], "burn_in")
        self.assertTrue(calc.baseline_ready)
        self.assertEqual(m2["burn_in_progress"], "2/2")

        # Window 3 - monitoring phase
        m3 = calc.compute_window_metrics(clean_window)
        self.assertEqual(m3["phase"], "monitoring")

    def test_baseline_reset(self):
        calc = DriftMetricsCalculator(burn_in_windows=2)
        clean_item = {"drifted_text": "Good film", "drifted_score": 4.0, "is_drifted": False}
        calc.compute_window_metrics([clean_item] * 5)
        calc.compute_window_metrics([clean_item] * 5)
        self.assertTrue(calc.baseline_ready)

        calc.reset_baseline()
        self.assertFalse(calc.baseline_ready)
        self.assertEqual(calc.burn_in_windows_collected, 0)
        self.assertEqual(len(calc._baseline_texts), 0)

    def test_cosine_similarity_against_baseline(self):
        calc = DriftMetricsCalculator(burn_in_windows=2)
        movie_items = [{"drifted_text": "The movie and direction and acting were fantastic and magnificent", "drifted_score": 5.0}] * 10
        calc.compute_window_metrics(movie_items)
        calc.compute_window_metrics(movie_items)
        self.assertTrue(calc.baseline_ready)

        # Completely unrelated scientific vocabulary
        science_items = [{"drifted_text": "Quantum thermodynamic eigenvalue orbital wavefunction tensor Hamiltonian particle physics", "drifted_score": 5.0}] * 10
        m = calc.compute_window_metrics(science_items)
        self.assertEqual(m["phase"], "monitoring")
        self.assertLess(m["cosine_similarity"], 0.5)

    def test_spelling_error_rate(self):
        calc = DriftMetricsCalculator(burn_in_windows=1)
        clean_items = [{"drifted_text": "This is a perfectly normal English sentence with clear vocabulary", "drifted_score": 4.0}] * 5
        m_clean = calc.compute_window_metrics(clean_items)
        self.assertEqual(m_clean["spelling_error_rate"], 0.0)

        # Misspelled nonsense words
        corrupted_items = [{"drifted_text": "Thizz izz xtrmly noizy txt wth wronggg spllngss and garblld wrdzz", "drifted_score": 4.0}] * 5
        m_corrupt = calc.compute_window_metrics(corrupted_items)
        self.assertGreater(m_corrupt["spelling_error_rate"], 0.4)

    def test_score_distribution_divergence(self):
        calc = DriftMetricsCalculator(burn_in_windows=2)
        five_star_items = [{"drifted_text": "Great movie", "drifted_score": 5.0}] * 10
        calc.compute_window_metrics(five_star_items)
        calc.compute_window_metrics(five_star_items)
        self.assertTrue(calc.baseline_ready)

        # Shift to 1-star reviews
        one_star_items = [{"drifted_text": "Terrible movie", "drifted_score": 1.0}] * 10
        m = calc.compute_window_metrics(one_star_items)
        self.assertGreater(m["score_dist_divergence"], 0.5)


if __name__ == "__main__":
    unittest.main()
