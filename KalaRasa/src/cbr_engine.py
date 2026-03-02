# src/cbr_engine.py  (v2 – Fase 1)
# Case-Based Reasoning Engine dengan:
#   - Feedback loop (apply_feedback, load/save weights)
#   - Case weighting yang persisten via JSON
#   - Grid search support (configurable WEIGHTS)

from __future__ import annotations


import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import TextPreprocessor


# ── RecipeCase (tidak berubah dari v1) ─────────────────────────────────────────

class RecipeCase:
    def __init__(self, data: Dict):
        self.recipe_id: int        = data["id"]
        self.nama: str             = data.get("nama", "")
        self.waktu_masak: int      = data.get("waktu_masak", 60)
        self.region: str           = (data.get("region") or "").lower()
        self.kategori: str         = (data.get("kategori") or "").lower()
        self.deskripsi: str        = data.get("deskripsi") or ""
        self.avg_rating: float     = float(data.get("avg_rating", 0))
        self.view_count: int       = int(data.get("view_count", 0))
        self.ingredients_main: List[str]  = [i.lower() for i in data.get("ingredients_main", [])]
        self.ingredients_all: List[str]   = [i.lower() for i in data.get("ingredients_all", [])]
        self.suitable_for: List[str] = [c.lower() for c in data.get("suitable_for", [])]
        self.not_suitable_for: List[str] = [c.lower() for c in data.get("not_suitable_for", [])]

    def to_text_representation(self) -> str:
        parts = []
        nama_clean = self.nama.lower()
        parts.extend([nama_clean] * 3)
        parts.extend(self.ingredients_main * 3)
        parts.extend(self.ingredients_all)
        if self.region:
            parts.extend([self.region] * 2)
        if self.kategori:
            parts.append(self.kategori)
        parts.extend(self.suitable_for)
        return " ".join(parts)

    def to_dict(self) -> Dict:
        return {
            "recipe_id":         self.recipe_id,
            "nama":              self.nama,
            "waktu_masak":       self.waktu_masak,
            "region":            self.region,
            "kategori":          self.kategori,
            "avg_rating":        self.avg_rating,
            "view_count":        self.view_count,
            "ingredients_main":  self.ingredients_main,
            "suitable_for":      self.suitable_for,
            "not_suitable_for":  self.not_suitable_for,
        }


# ── SimilarityCalculator ────────────────────────────────────────────────────────

class SimilarityCalculator:
    """
    Weighted similarity dengan bobot yang bisa dikonfigurasi dari luar.
    Default bobot bisa di-override via set_weights() (dari hasil grid search).
    """

    DEFAULT_WEIGHTS = {
        "text":       0.35,
        "ingredient": 0.30,
        "health":     0.20,
        "constraint": 0.15,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()

    def set_weights(self, weights: Dict[str, float]):
        """Update bobot similarity (dari hasil grid search)."""
        self.weights = weights.copy()
        self._validate_weights()

    def _validate_weights(self):
        required = {"text", "ingredient", "health", "constraint"}
        missing = required - set(self.weights.keys())
        if missing:
            raise ValueError(f"Weight keys kurang: {missing}")
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Total weight harus ~1.0, dapat {total:.4f}")

    def compute(
        self,
        query_vector: np.ndarray,
        case_vector: np.ndarray,
        query_entities: Dict,
        case: RecipeCase,
        case_weight: float = 1.0,   # ← v2: feedback weight modifier
    ) -> Tuple[float, Dict]:
        text_score       = float(cosine_similarity(query_vector, case_vector)[0][0])
        ingredient_score = self._ingredient_similarity(query_entities, case)
        health_score     = self._health_similarity(query_entities, case)
        constraint_score = self._constraint_similarity(query_entities, case)

        total = (
            self.weights["text"]       * text_score +
            self.weights["ingredient"] * ingredient_score +
            self.weights["health"]     * health_score +
            self.weights["constraint"] * constraint_score
        ) * case_weight  # ← v2: modifikasi skor dengan feedback weight

        # Clamp ke [0, 1]
        total = max(0.0, min(1.0, total))

        return total, {
            "text_similarity":       round(text_score, 4),
            "ingredient_similarity": round(ingredient_score, 4),
            "health_similarity":     round(health_score, 4),
            "constraint_similarity": round(constraint_score, 4),
            "case_weight":           round(case_weight, 4),
            "total":                 round(total, 4),
        }

    def _ingredient_similarity(self, query_entities: Dict, case: RecipeCase) -> float:
        query_ings = set(i.lower() for i in query_entities.get("ingredients", {}).get("main", []))
        if not query_ings:
            return 0.5
        case_ings = set(case.ingredients_main)
        if not case_ings:
            return 0.0
        intersection = query_ings & case_ings
        union        = query_ings | case_ings
        return len(intersection) / len(union) if union else 0.0

    def _health_similarity(self, query_entities: Dict, case: RecipeCase) -> float:
        conditions = [c.lower() for c in query_entities.get("health_conditions", [])]
        if not conditions:
            return 0.5
        for cond in conditions:
            if any(cond in nf for nf in case.not_suitable_for):
                return 0.0
        matched = sum(1 for cond in conditions if any(cond in sf for sf in case.suitable_for))
        return matched / len(conditions)

    def _constraint_similarity(self, query_entities: Dict, case: RecipeCase) -> float:
        score = 0.5
        time_limit = query_entities.get("time_constraint")
        if time_limit is not None:
            if case.waktu_masak <= time_limit:
                score += 0.3
            elif case.waktu_masak <= time_limit * 1.3:
                score += 0.1
            else:
                score -= 0.3
        region_query = (query_entities.get("region") or "").lower()
        if region_query and case.region:
            if region_query in case.region or case.region in region_query:
                score += 0.2
            else:
                score -= 0.1
        return max(0.0, min(1.0, score))


# ── CBREngine ───────────────────────────────────────────────────────────────────

class CBREngine:
    """
    CBR Engine v2.

    Penambahan Fase 1:
    1. case_weights: dict {recipe_id → float} – dipersist ke JSON
    2. apply_feedback(recipe_id, rating) – update weight dari feedback user
    3. load_weights() / save_weights() – persist ke disk
    4. set_weights() – update bobot similarity dari grid search
    """

    MIN_SIMILARITY = 0.10
    TOP_K          = 5

    # Batas atas/bawah feedback weight agar tidak overfit
    WEIGHT_MIN = 0.5
    WEIGHT_MAX = 2.0
    # Delta per feedback
    FEEDBACK_POSITIVE = +0.08   # asymmetric: reward < punishment
    FEEDBACK_NEGATIVE = -0.12

    def __init__(self, weights_path: str = "models/cbr_weights.json"):
        self.preprocessor    = TextPreprocessor()
        self.similarity_calc = SimilarityCalculator()

        self.cases: List[RecipeCase]           = []
        self.case_vectors: Optional[np.ndarray] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._cases_hash: str = ""

        # v2: per-case feedback weights
        self.weights_path  = weights_path
        self.case_weights: Dict[int, float] = {}   # {recipe_id: float}
        self.load_weights()

    # ── 1. Load & Index Cases ───────────────────────────────────────────────

    def load_cases(self, recipes_data: List[Dict]) -> int:
        new_hash = self._hash_cases(recipes_data)
        if new_hash == self._cases_hash and self.cases:
            return len(self.cases)
        self.cases       = [RecipeCase(r) for r in recipes_data]
        self._cases_hash = new_hash

        # Inisialisasi weight = 1.0 untuk resep baru yang belum punya weight
        for case in self.cases:
            if case.recipe_id not in self.case_weights:
                self.case_weights[case.recipe_id] = 1.0

        self._build_index()
        return len(self.cases)

    def _build_index(self):
        if not self.cases:
            return
        corpus = [c.to_text_representation() for c in self.cases]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_df=0.95,
            sublinear_tf=True, analyzer="word",
        )
        self.case_vectors = self.vectorizer.fit_transform(corpus).toarray()

    # ── 2. Retrieve ─────────────────────────────────────────────────────────

    def retrieve(self, query_text: str, entities: Dict, top_k: int = TOP_K) -> Dict:
        if not self.cases or self.vectorizer is None:
            return self._empty_result("No cases loaded")

        query_repr = self._build_query_representation(query_text, entities)
        query_vec  = self.vectorizer.transform([query_repr]).toarray()

        scored: List[Tuple[float, Dict, RecipeCase]] = []
        for i, case in enumerate(self.cases):
            case_vec     = self.case_vectors[i].reshape(1, -1)
            case_weight  = self.case_weights.get(case.recipe_id, 1.0)
            score, breakdown = self.similarity_calc.compute(
                query_vec, case_vec, entities, case, case_weight
            )

            # Hard filter
            if breakdown["health_similarity"] == 0.0 and entities.get("health_conditions"):
                continue
            avoid_ings = set(i.lower() for i in entities.get("ingredients", {}).get("avoid", []))
            if avoid_ings & set(case.ingredients_all):
                continue
            if score >= self.MIN_SIMILARITY:
                scored.append((score, breakdown, case))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = scored[:top_k]

        matched = []
        for rank, (score, breakdown, case) in enumerate(top_results, start=1):
            matched.append({
                "rank_position":    rank,
                "recipe_id":        case.recipe_id,
                "nama":             case.nama,
                "match_score":      round(score * 100, 2),
                "waktu_masak":      case.waktu_masak,
                "region":           case.region,
                "kategori":         case.kategori,
                "avg_rating":       case.avg_rating,
                "ingredients_main": case.ingredients_main,
                "suitable_for":     case.suitable_for,
                "score_breakdown":  breakdown,
            })

        query_hash = self._hash_query(query_text, entities)
        return {
            "matched_recipes":  matched,
            "total_candidates": len(scored),
            "query_hash":       query_hash,
            "cbr_metadata": {
                "total_cases":     len(self.cases),
                "algorithm":       "TF-IDF Cosine + Jaccard + Health Filter + Feedback Weights",
                "weights":         self.similarity_calc.weights,
                "query_repr":      query_repr,
            },
        }

    # ── 3. Feedback Loop (BARU v2) ──────────────────────────────────────────

    def apply_feedback(self, recipe_id: int, rating: int) -> Dict:
        """
        Update case weight berdasarkan feedback user.

        Args:
            recipe_id: ID resep yang diberi feedback
            rating:    1 = positif (👍), -1 = negatif (👎)

        Returns:
            {"recipe_id": int, "old_weight": float, "new_weight": float, "saved": bool}
        """
        old_weight = self.case_weights.get(recipe_id, 1.0)
        delta      = self.FEEDBACK_POSITIVE if rating > 0 else self.FEEDBACK_NEGATIVE
        new_weight = max(self.WEIGHT_MIN, min(self.WEIGHT_MAX, old_weight + delta))

        self.case_weights[recipe_id] = new_weight
        saved = self.save_weights()

        return {
            "recipe_id":  recipe_id,
            "old_weight": round(old_weight, 4),
            "new_weight": round(new_weight, 4),
            "delta":      round(delta, 4),
            "saved":      saved,
        }

    def apply_bulk_feedback(self, feedback_list: List[Dict]) -> List[Dict]:
        """
        Batch feedback update.
        feedback_list: [{"recipe_id": int, "rating": int}, ...]
        """
        results = []
        for item in feedback_list:
            result = self.apply_feedback(item["recipe_id"], item["rating"])
            results.append(result)
        return results

    # ── 4. Weight Persistence ────────────────────────────────────────────────

    def save_weights(self) -> bool:
        """Simpan case_weights ke JSON untuk persist antar restart."""
        try:
            os.makedirs(os.path.dirname(self.weights_path) or ".", exist_ok=True)
            payload = {
                "case_weights":       {str(k): v for k, v in self.case_weights.items()},
                "similarity_weights": self.similarity_calc.weights,
            }
            with open(self.weights_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return True
        except Exception as e:
            print(f"  ⚠ Gagal simpan weights: {e}")
            return False

    def load_weights(self) -> bool:
        """Load weights dari JSON."""
        try:
            with open(self.weights_path, encoding="utf-8") as f:
                payload = json.load(f)
            self.case_weights = {int(k): v for k, v in payload.get("case_weights", {}).items()}
            sim_w = payload.get("similarity_weights")
            if sim_w:
                self.similarity_calc.set_weights(sim_w)
            print(f"  ✓ Weights loaded: {len(self.case_weights)} cases from {self.weights_path}")
            return True
        except FileNotFoundError:
            print(f"  ℹ {self.weights_path} belum ada, menggunakan default weights")
            return False
        except Exception as e:
            print(f"  ⚠ Gagal load weights: {e}")
            return False

    # ── 5. Grid Search Interface (BARU v2) ──────────────────────────────────

    def set_similarity_weights(self, weights: Dict[str, float]):
        """
        Update bobot komponen similarity dari hasil grid search eksternal.
        Dipanggil setelah scripts/optimize_weights.py selesai.
        """
        self.similarity_calc.set_weights(weights)
        self.save_weights()
        print(f"  ✓ Similarity weights updated: {weights}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_query_representation(self, query_text: str, entities: Dict) -> str:
        parts = [query_text]
        for ing in entities.get("ingredients", {}).get("main", []):
            parts.extend([ing.lower()] * 3)
        for cond in entities.get("health_conditions", []):
            parts.extend([cond.lower()] * 2)
        region = (entities.get("region") or "").lower()
        if region:
            parts.extend([region] * 2)
        return " ".join(parts)

    @staticmethod
    def _hash_cases(data: List[Dict]) -> str:
        ids = sorted(str(d.get("id", "")) for d in data)
        return hashlib.md5("|".join(ids).encode()).hexdigest()

    @staticmethod
    def _hash_query(query: str, entities: Dict) -> str:
        payload = query + json.dumps(entities, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _empty_result(reason: str) -> Dict:
        return {
            "matched_recipes": [], "total_candidates": 0,
            "query_hash": "", "cbr_metadata": {"error": reason},
        }

    def get_stats(self) -> Dict:
        top_boosted = sorted(
            [(rid, w) for rid, w in self.case_weights.items() if w > 1.0],
            key=lambda x: x[1], reverse=True
        )[:5]
        top_penalized = sorted(
            [(rid, w) for rid, w in self.case_weights.items() if w < 1.0],
            key=lambda x: x[1]
        )[:5]
        return {
            "total_cases":     len(self.cases),
            "index_built":     self.vectorizer is not None,
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            "feedback_weights": {
                "total_modified": len([w for w in self.case_weights.values() if w != 1.0]),
                "top_boosted":    top_boosted,
                "top_penalized":  top_penalized,
            },
            "similarity_weights": self.similarity_calc.weights,
        }