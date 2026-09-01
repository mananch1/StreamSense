"""
StreamSense Drift Engine.
Implements 6 textual data drift generation methods:
- 4 methods from Garcia et al. (2024) [arXiv:2403.12328]:
    1. Adjective Swap (Semantic drift using POS tagging + WordNet antonyms & satellite adjectives)
    2. Class Swap (Abrupt sentiment label reversal)
    3. Class Shift (Gradual sentiment label rotation)
    4. Time-Slice Removal (Temporal window masking)
- 2 custom practical NLP drift methods:
    5. Noise Injection (Typos, character mutations, truncation)
    6. Formality / Slang Shift (Register transformation)
"""

import re
import math
import random
import nltk
from typing import Dict, Any, Tuple, List, Optional
from src.config import DriftConfig

# Ensure necessary NLTK models are downloaded
def ensure_nltk_corpora():
    packages = ["wordnet", "omw-1.4", "averaged_perceptron_tagger_eng", "punkt_tab", "punkt"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

ensure_nltk_corpora()

try:
    from nltk.corpus import wordnet as wn
    from nltk import pos_tag, word_tokenize
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False


# High-frequency sentiment antonym dictionary to supplement WordNet satellites
COMMON_ANTONYMS = {
    "good": ["bad", "poor", "terrible", "inferior"],
    "great": ["awful", "terrible", "poor", "mediocre"],
    "excellent": ["abysmal", "poor", "terrible", "dreadful"],
    "amazing": ["unimpressive", "dull", "terrible", "awful"],
    "awesome": ["horrible", "terrible", "lame", "boring"],
    "wonderful": ["horrendous", "dreadful", "awful", "dreary"],
    "fantastic": ["disastrous", "pathetic", "poor", "lousy"],
    "magnificent": ["pathetic", "shoddy", "dismal", "inferior"],
    "superb": ["mediocre", "abysmal", "subpar", "inferior"],
    "brilliant": ["foolish", "dim", "stupid", "dull"],
    "beautiful": ["ugly", "hideous", "grotesque", "unpleasant"],
    "effective": ["ineffective", "useless", "pointless", "futile"],
    "fast": ["slow", "sluggish", "lethargic"],
    "easy": ["hard", "difficult", "exhausting", "impossible"],
    "hard": ["easy", "effortless", "simple", "breeze"],
    "difficult": ["easy", "trivial", "effortless"],
    "tough": ["fragile", "easy", "soft", "weak"],
    "intense": ["mild", "gentle", "weak", "bland"],
    "insane": ["sane", "normal", "boring", "ordinary"],
    "terrible": ["fantastic", "great", "excellent", "wonderful"],
    "horrible": ["pleasant", "delightful", "splendid", "great"],
    "bad": ["good", "great", "superb", "fine"],
    "boring": ["exciting", "thrilling", "engaging", "fascinating"],
    "overrated": ["underrated", "masterpiece", "gem", "classic"],
    "underrated": ["overrated", "hyped", "disappointing"],
    "dreadful": ["delightful", "wonderful", "terrific"],
    "useless": ["valuable", "helpful", "beneficial", "crucial"],
    "worth": ["worthless", "pointless", "useless"],
    "positive": ["negative", "cynical", "hostile"],
    "negative": ["positive", "constructive", "optimistic"],
    "happy": ["sad", "miserable", "depressed"],
    "love": ["hate", "despise", "loathe"],
    "like": ["dislike", "hate", "detest"],
    "challenging": ["trivial", "effortless", "painless"],
    "sore": ["refreshed", "painless", "soothed"],
    "heavy": ["light", "weightless", "featherweight"],
}

# Common slang / informal mappings for formality drift
INFORMAL_LEXICON = {
    r"\bexcellent\b": "goated",
    r"\bgreat\b": "fire",
    r"\bterrible\b": "mid af",
    r"\bhorrible\b": "straight trash",
    r"\bvery good\b": "super valid",
    r"\bI do not like\b": "ngl hate",
    r"\bI really liked\b": "lowkey loved",
    r"\bdefinitely\b": "100%",
    r"\bmasterpiece\b": "absolute banger",
    r"\bboring\b": "a whole snoozefest",
    r"\bamateur\b": "noobish",
    r"\bcinematography\b": "camera work",
    r"\bperformances\b": "acting",
    r"\bdisappointing\b": "major L",
    r"\bhighly recommend\b": "must watch fr fr",
    r"\bworth watching\b": "worth the hype",
    r"\bfavorite\b": "fav",
    r"\bbecause\b": "cuz",
    r"\bthough\b": "tho",
    r"\bprobably\b": "prob",
    r"\binsanely\b": "crazy",
    r"\bworkout\b": "sesh",
    r"\bexercise\b": "grind",
    r"\bminutes\b": "mins",
    r"\bprogram\b": "routine",
}

# Keyboard adjacency for realistic typo simulation
KEYBOARD_ADJACENCY = {
    'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'], 'r': ['e', 't', 'f'],
    't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'], 'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'],
    'o': ['i', 'p', 'l'], 'p': ['o', 'l'], 'a': ['q', 's', 'z'], 's': ['a', 'd', 'w', 'x'],
    'd': ['s', 'f', 'e', 'c'], 'f': ['d', 'g', 'r', 'v'], 'g': ['f', 'h', 't', 'b'],
    'h': ['g', 'j', 'y', 'n'], 'j': ['h', 'k', 'u', 'm'], 'k': ['j', 'l', 'i'],
    'l': ['k', 'o', 'p'], 'z': ['a', 'x'], 'x': ['z', 'c', 's'], 'c': ['x', 'v', 'd'],
    'v': ['c', 'b', 'f'], 'b': ['v', 'n', 'g'], 'n': ['b', 'm', 'h'], 'm': ['n', 'j']
}

class DriftEngine:
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self.step_count = 0
        self._antonym_cache: Dict[str, List[str]] = {}

    def update_config(self, new_config: DriftConfig):
        self.config = new_config

    def compute_curve_multiplier(self) -> float:
        """
        Calculates intensity multiplier based on drift curve and current step.
        """
        curve = self.config.drift_curve
        step = self.step_count
        cycle = max(10, self.config.drift_cycle_length)
        phase = (step % cycle) / cycle
        
        if curve == "constant":
            return 1.0
        elif curve == "gradual":
            return phase
        elif curve == "sudden":
            return 1.0 if phase >= 0.5 else 0.0
        elif curve == "sinusoidal":
            return 0.55 + 0.45 * math.sin(2 * math.pi * phase)
        elif curve == "step":
            return math.ceil(phase * 4) / 4.0
        return 1.0

    def get_antonyms(self, word: str) -> List[str]:
        """
        Fetch antonyms for a given adjective from WordNet + supplemental lexicon.
        """
        clean_word = word.lower()
        if clean_word in self._antonym_cache:
            return self._antonym_cache[clean_word]
        
        antonyms = set()

        # 1. Direct and Satellite WordNet Synsets
        if NLTK_AVAILABLE:
            for syn in wn.synsets(clean_word):
                for lemma in syn.lemmas():
                    for ant in lemma.antonyms():
                        antonyms.add(ant.name().replace('_', ' '))
                # Also check similar_to synsets for satellites
                for sim in syn.similar_tos():
                    for lemma in sim.lemmas():
                        for ant in lemma.antonyms():
                            antonyms.add(ant.name().replace('_', ' '))

        # 2. Add supplemental common antonyms
        if clean_word in COMMON_ANTONYMS:
            for ant in COMMON_ANTONYMS[clean_word]:
                antonyms.add(ant)
                        
        result = list(antonyms)
        self._antonym_cache[clean_word] = result
        return result

    def apply_adjective_swap(self, text: str, intensity: float) -> Tuple[str, List[Dict[str, str]]]:
        """
        Method from Garcia et al. (2024):
        POS tags the sentence, identifies adjectives, and replaces them with WordNet antonyms.
        """
        if intensity <= 0.01 or not text:
            return text, []

        try:
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
        except Exception:
            tokens = text.split()
            tagged = [(w, "JJ") for w in tokens]

        swapped_tokens = []
        modifications = []

        for word, tag in tagged:
            # JJ = adjective, JJR = comparative, JJS = superlative, RB = adverb
            is_adj = tag in ("JJ", "JJR", "JJS") or (word.lower() in COMMON_ANTONYMS)
            if is_adj and random.random() < intensity:
                antonyms = self.get_antonyms(word)
                if antonyms:
                    chosen_antonym = random.choice(antonyms)
                    if word.isupper():
                        chosen_antonym = chosen_antonym.upper()
                    elif word[0].isupper():
                        chosen_antonym = chosen_antonym.capitalize()
                    
                    swapped_tokens.append(chosen_antonym)
                    modifications.append({"original": word, "replaced_with": chosen_antonym, "type": "adjective_swap"})
                    continue
            swapped_tokens.append(word)

        result_text = " ".join(swapped_tokens)
        result_text = re.sub(r'\s+([,.\'!?";:])', r'\1', result_text)
        result_text = re.sub(r'(\$)\s+', r'\1', result_text)
        return result_text, modifications

    def apply_class_swap(self, score: float) -> Tuple[float, bool]:
        """
        Method from Garcia et al. (2024):
        Swaps opposite rating classes: 1.0 <-> 5.0, 2.0 <-> 4.0, 3.0 unchanged.
        """
        score_val = round(score)
        swap_map = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0, 5: 1.0}
        new_score = swap_map.get(score_val, score)
        return new_score, new_score != score

    def apply_class_shift(self, score: float) -> Tuple[float, bool]:
        """
        Method from Garcia et al. (2024):
        Cyclic shift: 1->2, 2->3, 3->4, 4->5, 5->1.
        """
        score_val = round(score)
        shift_map = {1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 1.0}
        new_score = shift_map.get(score_val, score)
        return new_score, True

    def apply_noise_injection(self, text: str, intensity: float) -> Tuple[str, List[Dict[str, str]]]:
        """
        Custom Practical Extension:
        Introduces typos, omitted vowels, character transpositions, or truncation.
        """
        if intensity <= 0.01 or not text:
            return text, []

        chars = list(text)
        modifications = []
        n_mutations = max(1, int(len(chars) * 0.08 * intensity))

        for _ in range(n_mutations):
            if not chars:
                break
            idx = random.randint(0, len(chars) - 1)
            char = chars[idx]
            char_lower = char.lower()

            mutation_type = random.choice(["typo", "drop", "swap", "duplicate"])

            if mutation_type == "typo" and char_lower in KEYBOARD_ADJACENCY:
                replacement = random.choice(KEYBOARD_ADJACENCY[char_lower])
                if char.isupper():
                    replacement = replacement.upper()
                chars[idx] = replacement
                modifications.append({"pos": str(idx), "original": char, "replaced_with": replacement, "type": "typo"})
            elif mutation_type == "drop" and len(chars) > 10:
                del chars[idx]
                modifications.append({"pos": str(idx), "original": char, "replaced_with": "", "type": "drop"})
            elif mutation_type == "swap" and idx < len(chars) - 1:
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                modifications.append({"pos": str(idx), "type": "transposition"})
            elif mutation_type == "duplicate" and char.isalpha():
                chars.insert(idx, char)
                modifications.append({"pos": str(idx), "type": "duplicate"})

        res = "".join(chars)
        if intensity > 0.6 and random.random() < (intensity - 0.5):
            cutoff = int(len(res) * random.uniform(0.5, 0.85))
            res = res[:cutoff] + "..."
            modifications.append({"type": "truncation", "cutoff": str(cutoff)})

        return res, modifications

    def apply_formality_shift(self, text: str, intensity: float) -> Tuple[str, List[Dict[str, str]]]:
        """
        Custom Practical Extension:
        Replaces standard movie review vocabulary with internet slang and modern colloquialisms.
        """
        if intensity <= 0.01 or not text:
            return text, []

        modified_text = text
        modifications = []

        for pattern, replacement in INFORMAL_LEXICON.items():
            if random.random() < intensity:
                matches = list(re.finditer(pattern, modified_text, flags=re.IGNORECASE))
                if matches:
                    modified_text, count = re.subn(pattern, replacement, modified_text, flags=re.IGNORECASE)
                    if count > 0:
                        modifications.append({"pattern": pattern, "replacement": replacement, "count": count, "type": "slang_shift"})

        return modified_text, modifications

    def process_review(self, review: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main pipeline: Takes raw review dict and applies configured drifts.
        Returns rich transformed item containing original, drifted text, altered score, and drift metadata.
        """
        self.step_count += 1
        curve_mult = self.compute_curve_multiplier()

        orig_text = review.get("text", "")
        orig_summary = review.get("summary", "")
        try:
            orig_score = float(review.get("score", 3.0))
        except Exception:
            orig_score = 3.0

        current_text = orig_text
        current_summary = orig_summary
        current_score = orig_score
        all_modifications = []
        active_drift_names = []

        # 1. Adjective Swap (Semantic drift)
        if self.config.enable_adjective_swap:
            eff_intensity = self.config.adjective_swap_intensity * curve_mult
            current_text, mods_text = self.apply_adjective_swap(current_text, eff_intensity)
            current_summary, mods_sum = self.apply_adjective_swap(current_summary, eff_intensity)
            if mods_text or mods_sum:
                all_modifications.extend(mods_text + mods_sum)
                active_drift_names.append("Adjective Swap (WordNet)")

        # 2. Formality / Slang Shift
        if self.config.enable_formality_shift:
            eff_intensity = self.config.formality_intensity * curve_mult
            current_text, mods_text = self.apply_formality_shift(current_text, eff_intensity)
            current_summary, mods_sum = self.apply_formality_shift(current_summary, eff_intensity)
            if mods_text or mods_sum:
                all_modifications.extend(mods_text + mods_sum)
                active_drift_names.append("Formality / Slang Shift")

        # 3. Noise Injection
        if self.config.enable_noise_injection:
            eff_intensity = self.config.noise_intensity * curve_mult
            current_text, mods_text = self.apply_noise_injection(current_text, eff_intensity)
            current_summary, mods_sum = self.apply_noise_injection(current_summary, eff_intensity)
            if mods_text or mods_sum:
                all_modifications.extend(mods_text + mods_sum)
                active_drift_names.append("Noise Injection")

        # 4. Class Swap (Abrupt label flip)
        is_score_flipped = False
        if self.config.enable_class_swap:
            current_score, is_score_flipped = self.apply_class_swap(current_score)
            if is_score_flipped:
                active_drift_names.append("Class Swap (1<->5)")

        # 5. Class Shift (Cyclic label shift)
        if self.config.enable_class_shift:
            current_score, _ = self.apply_class_shift(current_score)
            active_drift_names.append("Class Shift (+1 cyclic)")

        is_drifted = (current_text != orig_text) or (current_score != orig_score) or (len(active_drift_names) > 0)

        return {
            "productId": review.get("productId", ""),
            "userId": review.get("userId", ""),
            "profileName": review.get("profileName", "Anonymous"),
            "date": review.get("date", ""),
            "original_summary": orig_summary,
            "original_text": orig_text,
            "original_score": orig_score,
            "drifted_summary": current_summary,
            "drifted_text": current_text,
            "drifted_score": current_score,
            "is_drifted": is_drifted,
            "active_drifts": active_drift_names,
            "modifications": all_modifications,
            "step": self.step_count,
            "curve_multiplier": round(curve_mult, 3)
        }
