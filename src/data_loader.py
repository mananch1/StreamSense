"""
StreamSense Data Loader.
Loads reviews from extracted CSV files or provides initial dataset streaming with cursor management.
"""

import os
import csv
import json
from typing import List, Dict, Any, Optional
from src.config import SAMPLE_CSV, TOP_MOVIE_CSV, META_JSON

# High-quality fallback reviews for instant zero-wait startup if dataset is still processing
SEED_REVIEWS = [
    {
        "productId": "B00006HAXW",
        "userId": "A1RSDE90N6RSZF",
        "profileName": "CinemaBuff_99",
        "helpfulness": "15/16",
        "score": "5.0",
        "time": "1042502400",
        "date": "2003-01-14 00:00:00",
        "summary": "An absolute masterpiece of cinematic history",
        "text": "The direction, storytelling, and poignant performances in this film are simply extraordinary. A breathtaking experience that stays with you long after the credits roll. Highly recommended for every genuine film lover."
    },
    {
        "productId": "B00006HAXW",
        "userId": "A2MNO78PQR",
        "profileName": "FilmCriticDan",
        "helpfulness": "8/9",
        "score": "4.0",
        "time": "1042588800",
        "date": "2003-01-15 00:00:00",
        "summary": "Great acting and compelling narrative",
        "text": "While the pacing slows down a bit in the second act, the character development and emotional climax more than make up for it. Excellent soundtrack and top-notch cinematography throughout."
    },
    {
        "productId": "B00006HAXW",
        "userId": "A3XYZ12345",
        "profileName": "MovieWatcher88",
        "helpfulness": "2/10",
        "score": "1.0",
        "time": "1042675200",
        "date": "2003-01-16 00:00:00",
        "summary": "Completely overrated and boring",
        "text": "I really do not understand the hype around this movie. The plot was predictable, the dialogue was dreadful, and the characters were completely unlikable. A total waste of two hours."
    },
    {
        "productId": "B00006HAXW",
        "userId": "A4UVW98765",
        "profileName": "Sarah_Reviews",
        "helpfulness": "5/5",
        "score": "5.0",
        "time": "1042761600",
        "date": "2003-01-17 00:00:00",
        "summary": "Pure perfection from start to finish",
        "text": "Stunning visuals, magnificent score, and an unforgettable climax. The lead actors gave brilliant performances that deserve every award imaginable."
    },
    {
        "productId": "B00006HAXW",
        "userId": "A5KLM45678",
        "profileName": "CasualViewer",
        "helpfulness": "3/4",
        "score": "3.0",
        "time": "1042848000",
        "date": "2003-01-18 00:00:00",
        "summary": "Decent watch but nothing revolutionary",
        "text": "It has some strong moments and decent special effects, but the storyline felt somewhat generic. Good for a weekend watch if you have nothing else to do."
    },
    {
        "productId": "B00006HAXW",
        "userId": "A6QWE78901",
        "profileName": "RetroFilmEnthusiast",
        "helpfulness": "11/12",
        "score": "4.0",
        "time": "1042934400",
        "date": "2003-01-19 00:00:00",
        "summary": "Solid direction and wonderfully atmospheric",
        "text": "The moody lighting and nuanced sound design create an immersive world. Strong supporting cast and well-written dialogue keep the tension alive from start to finish."
    },
    {
        "productId": "B00006HAXW",
        "userId": "A7ASD34567",
        "profileName": "DisappointedFan",
        "helpfulness": "7/8",
        "score": "2.0",
        "time": "1043020800",
        "date": "2003-01-20 00:00:00",
        "summary": "Fails to live up to expectations",
        "text": "A promising premise squandered by sloppy editing and illogical plot twists. The actors tried their best, but the script gave them very little meaningful material."
    }
]

class DataLoader:
    def __init__(self, mode: str = "sample"):
        self.mode = mode
        self.reviews: List[Dict[str, Any]] = []
        self.cursor = 0
        self.metadata: Dict[str, Any] = {}
        self.reload_data()

    def reload_data(self):
        target_path = SAMPLE_CSV if self.mode == "sample" else TOP_MOVIE_CSV
        
        # If the requested file doesn't exist, try the other or fall back to seeds
        if not os.path.exists(target_path):
            if os.path.exists(SAMPLE_CSV):
                target_path = SAMPLE_CSV
            elif os.path.exists(TOP_MOVIE_CSV):
                target_path = TOP_MOVIE_CSV
            else:
                target_path = None

        if target_path and os.path.exists(target_path):
            try:
                loaded = []
                with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("text"):
                            loaded.append(row)
                if loaded:
                    self.reviews = loaded
            except Exception as e:
                print(f"[DataLoader] Error reading CSV: {e}")

        # Fallback if CSV empty or not ready yet
        if not self.reviews:
            # Repeat seed reviews to create a smooth buffer of at least 100 items
            self.reviews = [dict(r) for r in (SEED_REVIEWS * 20)]

        # Load metadata if exists
        if os.path.exists(META_JSON):
            try:
                with open(META_JSON, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                pass

        if not self.metadata:
            self.metadata = {
                "target_asin": self.reviews[0].get("productId", "B00006HAXW") if self.reviews else "B00006HAXW",
                "total_reviews": len(self.reviews),
                "dataset_mode": self.mode
            }

    def get_next_review(self) -> Dict[str, Any]:
        """
        Gets the next review in chronological stream, cycling seamlessly.
        """
        if not self.reviews:
            self.reload_data()
        
        review = self.reviews[self.cursor]
        self.cursor = (self.cursor + 1) % len(self.reviews)
        return dict(review)

    def reset_cursor(self):
        self.cursor = 0

    def get_info(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "total_reviews_loaded": len(self.reviews),
            "current_cursor": self.cursor,
            "target_asin": self.metadata.get("target_asin", "B00006HAXW"),
            "score_distribution": self.metadata.get("score_distribution", {})
        }
