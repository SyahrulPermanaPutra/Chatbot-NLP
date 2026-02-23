# src/cbr_engine.py
# Case-Based Reasoning Engine – kala_rasa_jtv
#
# CBR Cycle:
#   1. RETRIEVE  – cari case serupa dari DB berdasarkan similarity
#   2. REUSE     – gunakan solusi case terbaik sebagai kandidat
#   3. REVISE    – sesuaikan dengan constraints user (health, waktu, region)
#   4. RETAIN    – simpan query baru sebagai case (via Laravel/DB)
#
# Arsitektur:
#   - Flask hanya menjalankan reasoning & similarity scoring
#   - Laravel melakukan query DB dan menyimpan hasil
#   - Redis meng-cache vector TF-IDF dan hasil similarity

from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessor import TextPreprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Case Representation (merepresentasikan 1 baris recipes + ingredients)
# ─────────────────────────────────────────────────────────────────────────────

class RecipeCase:
    """
    Representasi sebuah 'case' dalam CBR.

    Kolom sumber (dari DB kala_rasa_jtv):
        recipes.id, nama, waktu_masak, region, kategori, deskripsi,
        avg_rating, view_count
        ingredients (via recipe_ingredients JOIN ingredients)
        health_conditions (via recipe_suitability JOIN health_conditions)
    """

    def __init__(self, data: Dict):
        # ── Core fields (recipes table) ───────────────────────────────
        self.recipe_id: int        = data["id"]
        self.nama: str             = data.get("nama", "")
        self.waktu_masak: int      = data.get("waktu_masak", 60)
        self.region: str           = (data.get("region") or "").lower()
        self.kategori: str         = (data.get("kategori") or "").lower()
        self.deskripsi: str        = data.get("deskripsi") or ""
        self.avg_rating: float     = float(data.get("avg_rating", 0))
        self.view_count: int       = int(data.get("view_count", 0))

        # ── Ingredients (list of names – lowercase) ────────────────────
        self.ingredients_main: List[str]  = [
            i.lower() for i in data.get("ingredients_main", [])
        ]
        self.ingredients_all: List[str]   = [
            i.lower() for i in data.get("ingredients_all", [])
        ]

        # ── Health suitability ─────────────────────────────────────────
        # List kondisi yang COCOK (is_suitable=1)
        self.suitable_for: List[str] = [
            c.lower() for c in data.get("suitable_for", [])
        ]
        # List kondisi yang TIDAK COCOK (is_suitable=0)
        self.not_suitable_for: List[str] = [
            c.lower() for c in data.get("not_suitable_for", [])
        ]

    def to_text_representation(self) -> str:
        """
        Ubah case menjadi teks untuk TF-IDF vectorization.
        Bobot kata diperkuat dengan repetisi.
        """
        parts = []

        # Nama resep – bobot tinggi (×3)
        nama_clean = self.nama.lower()
        parts.extend([nama_clean] * 3)

        # Bahan utama – bobot tinggi (×3)
        parts.extend(self.ingredients_main * 3)

        # Semua bahan – bobot normal
        parts.extend(self.ingredients_all)

        # Region & kategori
        if self.region:
            parts.extend([self.region] * 2)
        if self.kategori:
            parts.append(self.kategori)

        # Kondisi kesehatan
        parts.extend(self.suitable_for)

        return " ".join(parts)

    def to_dict(self) -> Dict:
        """Serialisasi untuk JSON response ke Laravel."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Calculator
# ─────────────────────────────────────────────────────────────────────────────

class SimilarityCalculator:
    """
    Menghitung similarity antara query user dan recipe cases.

    Komponen similarity (weighted sum):
        1. text_sim       (0.35) – cosine similarity TF-IDF antara query & recipe text
        2. ingredient_sim (0.30) – Jaccard similarity bahan utama
        3. health_sim     (0.20) – kecocokan kondisi kesehatan
        4. constraint_sim (0.15) – waktu masak & region
    """

    WEIGHTS = {
        "text":       0.35,
        "ingredient": 0.30,
        "health":     0.20,
        "constraint": 0.15,
    }

    def compute(
        self,
        query_vector: np.ndarray,
        case_vector: np.ndarray,
        query_entities: Dict,
        case: RecipeCase,
    ) -> Tuple[float, Dict]:
        """
        Hitung similarity komprehensif.

        Returns:
            (total_score, breakdown_dict)
        """
        # 1. Text similarity (cosine)
        text_score = float(cosine_similarity(query_vector, case_vector)[0][0])

        # 2. Ingredient similarity (Jaccard)
        ingredient_score = self._ingredient_similarity(query_entities, case)

        # 3. Health condition similarity
        health_score = self._health_similarity(query_entities, case)

        # 4. Constraint similarity (waktu + region)
        constraint_score = self._constraint_similarity(query_entities, case)

        # Weighted sum
        total = (
            self.WEIGHTS["text"]       * text_score +
            self.WEIGHTS["ingredient"] * ingredient_score +
            self.WEIGHTS["health"]     * health_score +
            self.WEIGHTS["constraint"] * constraint_score
        )

        breakdown = {
            "text_similarity":       round(text_score, 4),
            "ingredient_similarity": round(ingredient_score, 4),
            "health_similarity":     round(health_score, 4),
            "constraint_similarity": round(constraint_score, 4),
            "total":                 round(total, 4),
        }

        return total, breakdown

    def _ingredient_similarity(self, query_entities: Dict, case: RecipeCase) -> float:
        """Jaccard similarity pada bahan utama."""
        query_ings = set(
            i.lower() for i in query_entities.get("ingredients", {}).get("main", [])
        )
        if not query_ings:
            return 0.5  # neutral jika user tidak menyebut bahan spesifik

        case_ings = set(case.ingredients_main)
        if not case_ings:
            return 0.0

        intersection = query_ings & case_ings
        union        = query_ings | case_ings
        return len(intersection) / len(union) if union else 0.0

    def _health_similarity(self, query_entities: Dict, case: RecipeCase) -> float:
        """
        Skor kondisi kesehatan:
         +1.0  jika semua kondisi user ada di suitable_for
         -1.0  jika ada kondisi user yang ada di not_suitable_for (HARD PENALTY)
          0.5  jika tidak ada kondisi yang disebutkan
        """
        conditions = [
            c.lower() for c in query_entities.get("health_conditions", [])
        ]
        if not conditions:
            return 0.5

        # Hard penalty – jika resep TIDAK cocok untuk kondisi user
        for cond in conditions:
            if any(cond in nf for nf in case.not_suitable_for):
                return 0.0   # eliminasi langsung

        # Positive match
        matched = sum(
            1 for cond in conditions
            if any(cond in sf for sf in case.suitable_for)
        )
        return matched / len(conditions)

    def _constraint_similarity(self, query_entities: Dict, case: RecipeCase) -> float:
        """Waktu masak + region similarity."""
        score = 0.5  # baseline

        # Waktu masak
        time_limit = query_entities.get("time_constraint")
        if time_limit is not None:
            if case.waktu_masak <= time_limit:
                score += 0.3
            elif case.waktu_masak <= time_limit * 1.3:
                score += 0.1   # sedikit melebihi – partial match
            else:
                score -= 0.3

        # Region
        region_query = (query_entities.get("region") or "").lower()
        if region_query and case.region:
            if region_query in case.region or case.region in region_query:
                score += 0.2
            else:
                score -= 0.1

        return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# CBR Engine – Core
# ─────────────────────────────────────────────────────────────────────────────

class CBREngine:
    """
    Main CBR Engine.

    Alur:
        1. receive_cases(recipes_data)  – Laravel mengirim data resep dari DB
        2. index_cases()                – bangun TF-IDF index (atau dari cache)
        3. retrieve(query, entities)    – RETRIEVE + REUSE + REVISE
        4. format_result()              – kembalikan structured JSON untuk Laravel
    """

    # Minimum similarity untuk masuk ke hasil
    MIN_SIMILARITY = 0.10
    # Jumlah kandidat yang dikembalikan
    TOP_K = 5

    def __init__(self):
        self.preprocessor  = TextPreprocessor()
        self.similarity_calc = SimilarityCalculator()

        self.cases: List[RecipeCase]      = []
        self.case_vectors: Optional[np.ndarray] = None
        self.vectorizer: Optional[TfidfVectorizer] = None

        # Cache hash dari case set untuk deteksi perubahan
        self._cases_hash: str = ""

    # ── 1. Load Cases ──────────────────────────────────────────────────────

    def load_cases(self, recipes_data: List[Dict]) -> int:
        """
        Terima data resep dari Laravel dan bangun case base.

        Format recipes_data (dikirim dari Laravel via /api/cbr/index):
        [
            {
                "id": 1,
                "nama": "Ayam Goreng Krispi",
                "waktu_masak": 30,
                "region": "Jawa",
                "kategori": "goreng",
                "deskripsi": "...",
                "avg_rating": 4.5,
                "view_count": 120,
                "ingredients_main": ["ayam"],
                "ingredients_all": ["ayam", "tepung", "bawang putih", "telur"],
                "suitable_for": ["diabetes", "hipertensi"],
                "not_suitable_for": ["kolesterol"]
            },
            ...
        ]
        """
        new_hash = self._hash_cases(recipes_data)
        if new_hash == self._cases_hash and self.cases:
            return len(self.cases)  # tidak perlu rebuild

        self.cases = [RecipeCase(r) for r in recipes_data]
        self._cases_hash = new_hash
        self._build_index()
        return len(self.cases)

    def _build_index(self):
        """Bangun TF-IDF matrix dari semua cases."""
        if not self.cases:
            return

        corpus = [c.to_text_representation() for c in self.cases]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),     # unigram + bigram
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,      # log TF untuk mengurangi dominasi kata frekuen
            analyzer="word",
        )
        self.case_vectors = self.vectorizer.fit_transform(corpus).toarray()

    # ── 2. Retrieve ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_text: str,
        entities: Dict,
        top_k: int = TOP_K,
    ) -> Dict:
        """
        RETRIEVE + REUSE + REVISE dalam satu pass.

        Args:
            query_text: teks query yang sudah dinormalisasi
            entities:   hasil NER (ingredients, health_conditions, time_constraint, region)
            top_k:      jumlah resep kandidat

        Returns:
            {
                "matched_recipes": [...],
                "cbr_metadata":    {...},
                "query_hash":      str,   # untuk caching di Redis/Laravel
            }
        """
        if not self.cases or self.vectorizer is None:
            return self._empty_result("No cases loaded")

        # ── Query vectorization ────────────────────────────────────────
        query_repr  = self._build_query_representation(query_text, entities)
        query_vec   = self.vectorizer.transform([query_repr]).toarray()

        # ── Compute similarity untuk semua cases ───────────────────────
        scored: List[Tuple[float, Dict, RecipeCase]] = []
        for i, case in enumerate(self.cases):
            case_vec = self.case_vectors[i].reshape(1, -1)
            score, breakdown = self.similarity_calc.compute(
                query_vec, case_vec, entities, case
            )

            # Hard filter: kondisi kesehatan tidak cocok → skip
            if breakdown["health_similarity"] == 0.0 and entities.get("health_conditions"):
                continue

            # Hard filter: avoid ingredients
            avoid_ings = set(
                i.lower() for i in entities.get("ingredients", {}).get("avoid", [])
            )
            if avoid_ings & set(case.ingredients_all):
                continue

            if score >= self.MIN_SIMILARITY:
                scored.append((score, breakdown, case))

        # ── Sort & take top-k ──────────────────────────────────────────
        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = scored[:top_k]

        # ── Format hasil untuk Laravel ─────────────────────────────────
        matched = []
        for rank, (score, breakdown, case) in enumerate(top_results, start=1):
            matched.append({
                "rank_position":  rank,
                "recipe_id":      case.recipe_id,
                "nama":           case.nama,
                "match_score":    round(score * 100, 2),   # 0-100 scale (matched_recipes.match_score)
                "waktu_masak":    case.waktu_masak,
                "region":         case.region,
                "kategori":       case.kategori,
                "avg_rating":     case.avg_rating,
                "ingredients_main": case.ingredients_main,
                "suitable_for":   case.suitable_for,
                "score_breakdown": breakdown,
            })

        query_hash = self._hash_query(query_text, entities)

        return {
            "matched_recipes":  matched,
            "total_candidates": len(scored),
            "query_hash":       query_hash,
            "cbr_metadata": {
                "total_cases":     len(self.cases),
                "algorithm":       "TF-IDF Cosine + Jaccard + Health Filter",
                "weights":         SimilarityCalculator.WEIGHTS,
                "query_repr":      query_repr,
            },
        }

    # ── 3. Helpers ──────────────────────────────────────────────────────────

    def _build_query_representation(self, query_text: str, entities: Dict) -> str:
        """
        Bangun representasi teks query yang diperkaya dengan entitas.
        Entitas bahan & kondisi diberi bobot tinggi (diulang).
        """
        parts = [query_text]

        ings = entities.get("ingredients", {})
        for ing in ings.get("main", []):
            parts.extend([ing.lower()] * 3)   # bobot ×3

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
            "matched_recipes":  [],
            "total_candidates": 0,
            "query_hash":       "",
            "cbr_metadata":     {"error": reason},
        }

    # ── 4. Case Stats (untuk monitoring) ────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "total_cases":    len(self.cases),
            "index_built":    self.vectorizer is not None,
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
        }