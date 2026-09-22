"""
StreamSense Experimental Evaluation Script.
============================================
Generates all quantitative data tables for the research paper by exercising
the drift engine, metrics calculator, and sentiment model pipeline offline.

Experiments:
  1. Per-method drift metric sensitivity analysis (Table I)
  2. Multi-intensity drift impact (Table II)  
  3. Model accuracy degradation under drift (Table III)
  4. Retraining recovery analysis (Table IV)
  5. Metric cross-sensitivity matrix (Table V)
  6. Drift curve dynamics comparison (Table VI)
"""

import sys
import os
import csv
import json
import random
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import DriftConfig
from src.drift_engine import DriftEngine
from src.drift_metrics import DriftMetricsCalculator
from src.sentiment_model import SentimentModel, score_to_sentiment
from src.data_loader import DataLoader

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiment_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_reviews():
    """Load reviews from the dataset."""
    loader = DataLoader(mode="sample")
    reviews = []
    seen = set()
    for _ in range(min(500, len(loader.reviews))):
        r = loader.get_next_review()
        key = r.get("text", "")[:100]
        if key not in seen:
            seen.add(key)
            reviews.append(r)
    print(f"[Data] Loaded {len(reviews)} unique reviews")
    return reviews


def run_burn_in(calc, reviews, n_windows=3, window_size=25):
    """Run burn-in phase with clean data to establish baseline."""
    engine_clean = DriftEngine(config=DriftConfig())
    for w in range(n_windows):
        items = []
        for i in range(window_size):
            idx = (w * window_size + i) % len(reviews)
            processed = engine_clean.process_review(reviews[idx])
            items.append(processed)
        calc.compute_window_metrics(items)
    assert calc.baseline_ready, "Baseline should be ready after burn-in"


def experiment_1_per_method_sensitivity(reviews):
    """
    Table I: Per-Method Drift Metric Sensitivity Analysis.
    Applies each drift method individually at intensity=0.7 and records all metrics.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Per-Method Drift Metric Sensitivity Analysis")
    print("="*80)

    methods = [
        ("No Drift (Clean)", DriftConfig()),
        ("Adjective Swap", DriftConfig(enable_adjective_swap=True, adjective_swap_intensity=0.7)),
        ("Class Swap", DriftConfig(enable_class_swap=True)),
        ("Class Shift", DriftConfig(enable_class_shift=True)),
        ("Noise Injection", DriftConfig(enable_noise_injection=True, noise_intensity=0.7)),
        ("Formality Shift", DriftConfig(enable_formality_shift=True, formality_intensity=0.7)),
    ]

    results = []
    window_size = 25
    n_monitoring_windows = 3

    for method_name, config in methods:
        print(f"\n  Testing: {method_name}...")
        calc = DriftMetricsCalculator(burn_in_windows=3)
        run_burn_in(calc, reviews, n_windows=3, window_size=window_size)

        engine = DriftEngine(config=config)
        
        # Average over multiple monitoring windows
        all_metrics = []
        for w in range(n_monitoring_windows):
            items = []
            for i in range(window_size):
                idx = ((3 + w) * window_size + i) % len(reviews)
                processed = engine.process_review(reviews[idx])
                items.append(processed)
            m = calc.compute_window_metrics(items)
            all_metrics.append(m)
        
        # Average the metrics
        avg = {}
        metric_keys = [
            "cosine_similarity", "vocab_overlap", "sentiment_kl_divergence",
            "sentiment_js_divergence", "sentiment_wasserstein_distance",
            "spelling_error_rate", "score_dist_divergence", "drift_magnitude_pct"
        ]
        for key in metric_keys:
            vals = [m[key] for m in all_metrics]
            avg[key] = round(float(np.mean(vals)), 4)
        
        row = {"method": method_name}
        row.update(avg)
        results.append(row)
        
        print(f"    Cosine={avg['cosine_similarity']:.4f}  Vocab={avg['vocab_overlap']:.4f}  "
              f"KL={avg['sentiment_kl_divergence']:.4f}  JSD={avg['sentiment_js_divergence']:.4f}  "
              f"W1={avg['sentiment_wasserstein_distance']:.4f}  SER={avg['spelling_error_rate']:.4f}  "
              f"ScoreDiv={avg['score_dist_divergence']:.4f}  Composite={avg['drift_magnitude_pct']:.1f}%")

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "table1_per_method_sensitivity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + metric_keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  -> Saved: {csv_path}")
    return results


def experiment_2_multi_intensity(reviews):
    """
    Table II: Multi-Intensity Drift Impact.
    Tests Adjective Swap, Noise Injection, and Formality Shift at varying intensities.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Multi-Intensity Drift Impact")
    print("="*80)

    intensities = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    window_size = 25

    for method_name, config_fn in [
        ("Adjective Swap", lambda i: DriftConfig(enable_adjective_swap=True, adjective_swap_intensity=i)),
        ("Noise Injection", lambda i: DriftConfig(enable_noise_injection=True, noise_intensity=i)),
        ("Formality Shift", lambda i: DriftConfig(enable_formality_shift=True, formality_intensity=i)),
    ]:
        for intensity in intensities:
            config = config_fn(intensity)
            calc = DriftMetricsCalculator(burn_in_windows=3)
            run_burn_in(calc, reviews, n_windows=3, window_size=window_size)

            engine = DriftEngine(config=config)
            items = []
            for i in range(window_size):
                idx = (75 + i) % len(reviews)
                processed = engine.process_review(reviews[idx])
                items.append(processed)
            m = calc.compute_window_metrics(items)

            row = {
                "method": method_name,
                "intensity": intensity,
                "cosine_similarity": round(m["cosine_similarity"], 4),
                "vocab_overlap": round(m["vocab_overlap"], 4),
                "sentiment_js_divergence": round(m["sentiment_js_divergence"], 4),
                "spelling_error_rate": round(m["spelling_error_rate"], 4),
                "drift_magnitude_pct": round(m["drift_magnitude_pct"], 1),
            }
            results.append(row)
            print(f"  {method_name} @ {intensity:.2f}: Cosine={row['cosine_similarity']:.4f}  "
                  f"SER={row['spelling_error_rate']:.4f}  Composite={row['drift_magnitude_pct']:.1f}%")

    csv_path = os.path.join(OUTPUT_DIR, "table2_multi_intensity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "intensity", "cosine_similarity",
                                                "vocab_overlap", "sentiment_js_divergence",
                                                "spelling_error_rate", "drift_magnitude_pct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  -> Saved: {csv_path}")
    return results


def experiment_3_model_degradation(reviews):
    """
    Table III: Sentiment Model Accuracy Degradation Under Drift.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: Model Accuracy Degradation Under Drift")
    print("="*80)

    model = SentimentModel()
    train_texts = [r.get("text", "") for r in reviews[:100]]
    train_scores = [float(r.get("score", 3.0)) for r in reviews[:100]]
    train_res = model.train(train_texts, train_scores, version="v1.0")
    print(f"  Model v1.0 trained: accuracy={train_res['train_accuracy']:.4f} on {train_res['sample_count']} samples")

    drift_configs = [
        ("No Drift", DriftConfig()),
        ("Adjective Swap (0.7)", DriftConfig(enable_adjective_swap=True, adjective_swap_intensity=0.7)),
        ("Class Swap", DriftConfig(enable_class_swap=True)),
        ("Noise Injection (0.7)", DriftConfig(enable_noise_injection=True, noise_intensity=0.7)),
        ("Formality Shift (0.7)", DriftConfig(enable_formality_shift=True, formality_intensity=0.7)),
        ("Combined (Adj+Noise)", DriftConfig(
            enable_adjective_swap=True, adjective_swap_intensity=0.7,
            enable_noise_injection=True, noise_intensity=0.5
        )),
        ("All Methods", DriftConfig(
            enable_adjective_swap=True, adjective_swap_intensity=0.5,
            enable_class_swap=True,
            enable_noise_injection=True, noise_intensity=0.5,
            enable_formality_shift=True, formality_intensity=0.5
        )),
    ]

    results = []
    window_size = 25
    n_windows = 5

    for drift_name, config in drift_configs:
        model_copy = SentimentModel()
        model_copy.train(train_texts, train_scores, version="v1.0")
        engine = DriftEngine(config=config)

        window_accuracies = []
        for w in range(n_windows):
            for i in range(window_size):
                idx = (100 + w * window_size + i) % len(reviews)
                processed = engine.process_review(reviews[idx])
                model_copy.evaluate_item(
                    text=processed["drifted_text"],
                    score=float(processed["drifted_score"])
                )
            perf = model_copy.flush_window_metrics()
            window_accuracies.append(perf["model_window_accuracy"])

        avg_acc = round(float(np.mean(window_accuracies)), 4)
        min_acc = round(float(np.min(window_accuracies)), 4)
        row = {
            "drift_method": drift_name,
            "train_accuracy": train_res["train_accuracy"],
            "window_1_acc": round(window_accuracies[0], 4),
            "window_2_acc": round(window_accuracies[1], 4),
            "window_3_acc": round(window_accuracies[2], 4),
            "window_4_acc": round(window_accuracies[3], 4),
            "window_5_acc": round(window_accuracies[4], 4),
            "avg_accuracy": avg_acc,
            "min_accuracy": min_acc,
            "accuracy_drop": round(train_res["train_accuracy"] - avg_acc, 4),
        }
        results.append(row)
        print(f"  {drift_name}: avg_acc={avg_acc:.4f}  min={min_acc:.4f}  "
              f"drop={row['accuracy_drop']:.4f}  windows={[round(x,3) for x in window_accuracies]}")

    csv_path = os.path.join(OUTPUT_DIR, "table3_model_degradation.csv")
    fieldnames = ["drift_method", "train_accuracy", "window_1_acc", "window_2_acc",
                   "window_3_acc", "window_4_acc", "window_5_acc", "avg_accuracy",
                   "min_accuracy", "accuracy_drop"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  -> Saved: {csv_path}")
    return results


def experiment_4_retraining_recovery(reviews):
    """
    Table IV: Retraining Recovery Analysis.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: Retraining Recovery Analysis")
    print("="*80)

    results = []
    window_size = 25

    drift_configs = [
        ("Adjective Swap (0.7)", DriftConfig(enable_adjective_swap=True, adjective_swap_intensity=0.7)),
        ("Class Swap", DriftConfig(enable_class_swap=True)),
        ("Noise Injection (0.7)", DriftConfig(enable_noise_injection=True, noise_intensity=0.7)),
        ("Combined (Adj+Class+Noise)", DriftConfig(
            enable_adjective_swap=True, adjective_swap_intensity=0.5,
            enable_class_swap=True,
            enable_noise_injection=True, noise_intensity=0.5
        )),
    ]

    for drift_name, config in drift_configs:
        model = SentimentModel()
        engine_clean = DriftEngine(config=DriftConfig())
        engine_drift = DriftEngine(config=config)

        # Phase 1: Train on clean data
        train_texts = [r.get("text", "") for r in reviews[:80]]
        train_scores = [float(r.get("score", 3.0)) for r in reviews[:80]]
        model.train(train_texts, train_scores, version="v1.0")

        # Phase 2: Measure clean accuracy (2 windows)
        clean_accs = []
        for w in range(2):
            for i in range(window_size):
                idx = (80 + w * window_size + i) % len(reviews)
                processed = engine_clean.process_review(reviews[idx])
                model.evaluate_item(processed["drifted_text"], float(processed["drifted_score"]))
            perf = model.flush_window_metrics()
            clean_accs.append(perf["model_window_accuracy"])

        # Phase 3: Apply drift (3 windows), collect drifted data for retraining
        drift_accs = []
        retrain_texts = []
        retrain_scores = []
        for w in range(3):
            for i in range(window_size):
                idx = (130 + w * window_size + i) % len(reviews)
                processed = engine_drift.process_review(reviews[idx])
                model.evaluate_item(processed["drifted_text"], float(processed["drifted_score"]))
                retrain_texts.append(processed["drifted_text"])
                retrain_scores.append(float(processed["drifted_score"]))
            perf = model.flush_window_metrics()
            drift_accs.append(perf["model_window_accuracy"])

        # Phase 4: Retrain on accumulated drifted data
        model.train(retrain_texts, retrain_scores, version="v2.0")

        # Phase 5: Measure accuracy on continued drifted stream (2 windows)
        recovery_accs = []
        for w in range(2):
            for i in range(window_size):
                idx = (205 + w * window_size + i) % len(reviews)
                processed = engine_drift.process_review(reviews[idx])
                model.evaluate_item(processed["drifted_text"], float(processed["drifted_score"]))
            perf = model.flush_window_metrics()
            recovery_accs.append(perf["model_window_accuracy"])

        row = {
            "drift_method": drift_name,
            "clean_avg_accuracy": round(float(np.mean(clean_accs)), 4),
            "drifted_avg_accuracy": round(float(np.mean(drift_accs)), 4),
            "recovery_avg_accuracy": round(float(np.mean(recovery_accs)), 4),
            "degradation": round(float(np.mean(clean_accs) - np.mean(drift_accs)), 4),
            "recovery_gain": round(float(np.mean(recovery_accs) - np.mean(drift_accs)), 4),
        }
        results.append(row)
        print(f"  {drift_name}:")
        print(f"    Clean: {row['clean_avg_accuracy']:.4f}  Drifted: {row['drifted_avg_accuracy']:.4f}  "
              f"Recovered: {row['recovery_avg_accuracy']:.4f}  "
              f"Degradation: -{row['degradation']:.4f}  Recovery: +{row['recovery_gain']:.4f}")

    csv_path = os.path.join(OUTPUT_DIR, "table4_retraining_recovery.csv")
    fieldnames = ["drift_method", "clean_avg_accuracy", "drifted_avg_accuracy",
                   "recovery_avg_accuracy", "degradation", "recovery_gain"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  -> Saved: {csv_path}")
    return results


def experiment_5_metric_cross_sensitivity(reviews):
    """
    Table V: Metric Cross-Sensitivity Matrix.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 5: Metric Cross-Sensitivity Matrix")
    print("="*80)

    methods = [
        ("Adjective Swap", DriftConfig(enable_adjective_swap=True, adjective_swap_intensity=0.8)),
        ("Class Swap", DriftConfig(enable_class_swap=True)),
        ("Class Shift", DriftConfig(enable_class_shift=True)),
        ("Noise Injection", DriftConfig(enable_noise_injection=True, noise_intensity=0.8)),
        ("Formality Shift", DriftConfig(enable_formality_shift=True, formality_intensity=0.8)),
    ]

    metric_keys = [
        "cosine_similarity", "vocab_overlap", "sentiment_js_divergence",
        "spelling_error_rate", "score_dist_divergence"
    ]

    # First get clean baseline values
    calc_clean = DriftMetricsCalculator(burn_in_windows=3)
    run_burn_in(calc_clean, reviews, n_windows=3, window_size=25)
    engine_clean = DriftEngine(config=DriftConfig())
    clean_items = [engine_clean.process_review(reviews[i % len(reviews)]) for i in range(25)]
    clean_metrics = calc_clean.compute_window_metrics(clean_items)

    results = []
    for method_name, config in methods:
        calc = DriftMetricsCalculator(burn_in_windows=3)
        run_burn_in(calc, reviews, n_windows=3, window_size=25)
        engine = DriftEngine(config=config)

        items = [engine.process_review(reviews[(75 + i) % len(reviews)]) for i in range(25)]
        m = calc.compute_window_metrics(items)

        deltas = {}
        for key in metric_keys:
            if key in ["cosine_similarity", "vocab_overlap"]:
                delta = clean_metrics[key] - m[key]
            else:
                delta = m[key] - clean_metrics[key]
            deltas[key] = max(0.0, round(delta, 4))

        row = {"method": method_name}
        row.update(deltas)
        results.append(row)

        formatted = "  ".join(f"{k}={deltas[k]:.4f}" for k in metric_keys)
        print(f"  {method_name}: {formatted}")

    csv_path = os.path.join(OUTPUT_DIR, "table5_cross_sensitivity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + metric_keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  -> Saved: {csv_path}")
    return results


def experiment_6_drift_curves(reviews):
    """
    Table VI: Drift Curve Dynamics Comparison.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 6: Drift Curve Dynamics Comparison")
    print("="*80)

    curves = ["constant", "gradual", "sudden", "sinusoidal", "step"]
    results = []
    window_size = 20
    n_windows = 8

    for curve in curves:
        config = DriftConfig(
            enable_adjective_swap=True,
            adjective_swap_intensity=0.7,
            enable_noise_injection=True,
            noise_intensity=0.5,
            drift_curve=curve,
            drift_cycle_length=n_windows * window_size
        )

        calc = DriftMetricsCalculator(burn_in_windows=2)
        run_burn_in(calc, reviews, n_windows=2, window_size=window_size)
        engine = DriftEngine(config=config)

        window_scores = []
        for w in range(n_windows):
            items = []
            for i in range(window_size):
                idx = (40 + w * window_size + i) % len(reviews)
                processed = engine.process_review(reviews[idx])
                items.append(processed)
            m = calc.compute_window_metrics(items)
            window_scores.append(round(m["drift_magnitude_pct"], 1))

        row = {"curve": curve}
        for w_idx, score in enumerate(window_scores):
            row[f"window_{w_idx+1}"] = score
        row["avg_drift"] = round(float(np.mean(window_scores)), 1)
        row["max_drift"] = round(float(np.max(window_scores)), 1)
        row["std_drift"] = round(float(np.std(window_scores)), 1)
        results.append(row)

        print(f"  {curve:12s}: windows={window_scores}  avg={row['avg_drift']:.1f}%  max={row['max_drift']:.1f}%")

    csv_path = os.path.join(OUTPUT_DIR, "table6_drift_curves.csv")
    fieldnames = ["curve"] + [f"window_{i+1}" for i in range(n_windows)] + ["avg_drift", "max_drift", "std_drift"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  -> Saved: {csv_path}")
    return results


def main():
    print("StreamSense Experimental Evaluation")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")

    reviews = load_reviews()

    t1 = experiment_1_per_method_sensitivity(reviews)
    t2 = experiment_2_multi_intensity(reviews)
    t3 = experiment_3_model_degradation(reviews)
    t4 = experiment_4_retraining_recovery(reviews)
    t5 = experiment_5_metric_cross_sensitivity(reviews)
    t6 = experiment_6_drift_curves(reviews)

    # Save summary JSON
    summary = {
        "experiment_1_rows": len(t1),
        "experiment_2_rows": len(t2),
        "experiment_3_rows": len(t3),
        "experiment_4_rows": len(t4),
        "experiment_5_rows": len(t5),
        "experiment_6_rows": len(t6),
    }
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
