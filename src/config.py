"""
Configuration parameters for StreamSense.
"""

import os
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_CSV = os.path.join(DATA_DIR, "sample_reviews.csv")
TOP_MOVIE_CSV = os.path.join(DATA_DIR, "top_movie_reviews.csv")
META_JSON = os.path.join(DATA_DIR, "dataset_meta.json")

class DriftConfig(BaseModel):
    # Active drift modes
    enable_adjective_swap: bool = Field(False, description="Semantic drift via WordNet antonyms")
    enable_class_swap: bool = Field(False, description="Abrupt reversal of sentiment classes (e.g. 1<->5, 2<->4)")
    enable_class_shift: bool = Field(False, description="Gradual cyclic shift of sentiment classes (+1)")
    enable_time_slice_removal: bool = Field(False, description="Simulate temporal gaps / missing epochs")
    enable_noise_injection: bool = Field(False, description="Typos, missing characters, word truncation")
    enable_formality_shift: bool = Field(False, description="Slang, colloquialisms, internet text style")
    
    # Intensities (0.0 to 1.0)
    adjective_swap_intensity: float = Field(0.5, ge=0.0, le=1.0)
    noise_intensity: float = Field(0.3, ge=0.0, le=1.0)
    formality_intensity: float = Field(0.5, ge=0.0, le=1.0)
    
    # Drift dynamics curve
    drift_curve: Literal["constant", "gradual", "sudden", "sinusoidal", "step"] = "constant"
    drift_step: int = Field(0, description="Current time-step in stream")
    drift_cycle_length: int = Field(100, description="Cycle length for periodic/gradual drifts")

class FeedConfig(BaseModel):
    messages_per_second: float = Field(2.0, ge=0.1, le=50.0)
    window_message_count: int = Field(25, ge=5, le=500, description="Window size N (messages)")
    window_time_seconds: float = Field(15.0, ge=1.0, le=300.0, description="Window timeout T (seconds)")
    dataset_mode: Literal["sample", "full"] = "sample"
    is_running: bool = False
