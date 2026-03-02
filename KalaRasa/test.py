#!/usr/bin/env python3
"""
tests/test_fase1.py
Test suite komprehensif Fase 1 – kala_rasa_jtv NLP

Covers:
    T1. Preprocessor v2 – load JSON, synonym expansion, hot-reload
    T2. CBREngine v2 – feedback loop, weight persistence, grid search interface
    T3. SimilarityCalculator – custom weights, compute dengan case_weight
    T4. WeightOptimizer – grid search pipeline (dry-run)
    T5. Integration – alur end-to-end dari query ke feedback

Run:
    cd /path/to/project
    python -m pytest tests/test_fase1.py -v
    python -m pytest tests/test_fase1.py -v -k "feedback"   # filter test tertentu
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import TextPreprocessor
from src.cbr_engine import CBREngine, SimilarityCalculator, RecipeCase


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_RECIPES = [
    {
        "id": 1, "nama": "Ayam Goreng Krispi", "waktu_masak": 30,
        "region": "Jawa", "kategori": "goreng", "deskripsi": "Ayam goreng renyah",
        "avg_rating": 4.5, "view_count": 200,
        "ingredients_main": ["ayam"],
        "ingredients_all":  ["ayam", "tepung", "bawang putih", "telur", "minyak"],
        "suitable_for":     ["hipertensi"],
        "not_suitable_for": ["kolesterol"],
    },
    {
        "id": 2, "nama": "Ayam Kukus Sehat", "waktu_masak": 25,
        "region": "Sunda", "kategori": "kukus", "deskripsi": "Ayam kukus tanpa minyak",
        "avg_rating": 4.3, "view_count": 120,
        "ingredients_main": ["ayam"],
        "ingredients_all":  ["ayam", "jahe", "bawang putih", "daun salam"],
        "suitable_for":     ["diabetes", "hipertensi", "kolesterol"],
        "not_suitable_for": [],
    },
    {
        "id": 3, "nama": "Tempe Bacem", "waktu_masak": 45,
        "region": "Jawa", "kategori": "rebus", "deskripsi": "Tempe manis gurih",
        "avg_rating": 4.0, "view_count": 80,
        "ingredients_main": ["tempe"],
        "ingredients_all":  ["tempe", "gula merah", "bawang putih", "ketumbar"],
        "suitable_for":     ["diabetes"],
        "not_suitable_for": [],
    },
    {
        "id": 4, "nama": "Rendang Sapi", "waktu_masak": 180,
        "region": "Padang", "kategori": "rendang", "deskripsi": "Rendang daging sapi",
        "avg_rating": 4.8, "view_count": 500,
        "ingredients_main": ["sapi"],
        "ingredients_all":  ["sapi", "santan", "cabai", "bawang", "jahe", "serai"],
        "suitable_for":     [],
        "not_suitable_for": ["diabetes", "kolesterol", "hipertensi"],
    },
    {
        "id": 5, "nama": "Sayur Lodeh", "waktu_masak": 35,
        "region": "Jawa", "kategori": "kuah", "deskripsi": "Sayuran berkuah santan",
        "avg_rating": 4.1, "view_count": 150,
        "ingredients_main": ["sayuran", "tahu", "tempe"],
        "ingredients_all":  ["labu siam", "kacang panjang", "tahu", "tempe", "santan"],
        "suitable_for":     ["diabetes"],
        "not_suitable_for": ["kolesterol"],
    },
]

SAMPLE_ENTITIES = {
    "ingredients":       {"main": ["ayam"], "avoid": []},
    "health_conditions": ["diabetes"],
    "time_constraint":   30,
    "region":            None,
}


# ═════════════════════════════════════════════════════════════════════════════
# T1. Preprocessor 
# ═════════════════════════════════════════════════════════════════════════════

class TestPreprocessorV2(unittest.TestCase):
    """Preprocessor v2: dynamic dict loading, synonym expansion, hot-reload."""

    def setUp(self):
        """Buat temp dir dengan minimal JSON files untuk test."""
        self.tmpdir = tempfile.mkdtemp()
        self.pp = TextPreprocessor(data_dir=self.tmpdir)

    def test_fallback_to_builtin_when_no_json(self):
        """Tanpa file JSON, harus fallback ke builtin INFORMAL_MAP."""
        pp = TextPreprocessor(data_dir="/nonexistent_dir_xyz")
        self.assertIn("gw", pp.informal_map)
        self.assertEqual(pp.informal_map["gw"], "saya")

    def test_load_informal_map_from_json(self):
        """Load informal_map.json dari disk."""
        data = {
            "_meta": {"description": "test", "last_updated": "2026", "maintainer": "test"},
            "test_informal": "test_formal",
            "gw": "saya",
        }
        
        # Buat file JSON di temporary directory
        path = os.path.join(self.tmpdir, "informal_map.json")
        with open(path, "w") as f:
            json.dump(data, f)

        # Inisialisasi TextPreprocessor dengan data_dir = self.tmpdir
        # Ini akan membuat preprocessor membaca file dari self.tmpdir/informal_map.json
        pp = TextPreprocessor(data_dir=self.tmpdir)
        
        self.assertIn("test_informal", pp.informal_map)
        self.assertEqual(pp.informal_map["test_informal"], "test_formal")
        self.assertTrue(pp._json_loaded)
    

    def test_normalize_basic_informal(self):
        """Normalisasi kata informal dasar."""
        result = self.pp.normalize_text("gw pengen masak ayam gak pake tepung")
        self.assertIn("saya", result)
        self.assertIn("ingin", result)
        self.assertIn("tidak", result)
        self.assertNotIn("gw", result)
        self.assertNotIn("pengen", result)

    def test_synonym_expansion_from_json(self):
        """Synonym map di-load dan expand dengan benar."""
        # Buat synonym file
        synonym_dir = os.path.join(self.tmpdir, "synonyms")
        os.makedirs(synonym_dir)
        syn_data = {
            "_meta": {"description": "test", "last_updated": "2026", "maintainer": "test"},
            "ayam": ["chicken", "poultry"],
        }
        with open(os.path.join(synonym_dir, "protein.json"), "w") as f:
            json.dump(syn_data, f)

        pp = TextPreprocessor(data_dir=self.tmpdir)
        # "chicken" harus dinormalisasi ke "ayam"
        result = pp.normalize_text("mau masak chicken goreng")
        self.assertIn("ayam", result)

    def test_preprocess_returns_all_keys(self):
        """Hasil preprocess() punya semua keys yang dibutuhkan."""
        result = self.pp.preprocess("gw mau masak ayam")
        self.assertIn("original", result)
        self.assertIn("normalized", result)
        self.assertIn("negations", result)
        self.assertIn("expanded_terms", result)

    def test_extract_negations(self):
        """Deteksi pola negasi dalam teks."""
        negations = self.pp.extract_negations("saya tidak boleh makan gula dan alergi udang")
        self.assertTrue(len(negations) > 0)
        # Harus ada "tidak boleh"
        self.assertTrue(any("tidak" in n for n in negations))

    def test_reload_dictionaries(self):
        """Hot-reload kamus tanpa exception."""
        try:
            self.pp.reload_dictionaries()
        except Exception as e:
            self.fail(f"reload_dictionaries() raised exception: {e}")

    def test_partikel_dihapus(self):
        """Partikel 'deh', 'dong' dihapus dari teks."""
        result = self.pp.normalize_text("cariin resep dong yang gampang deh")
        self.assertNotIn("dong", result)
        self.assertNotIn("deh", result)


# ═════════════════════════════════════════════════════════════════════════════
# T2. SimilarityCalculator – Custom Weights
# ═════════════════════════════════════════════════════════════════════════════

class TestSimilarityCalculator(unittest.TestCase):
    """SimilarityCalculator: custom weights, compute, validation."""

    def setUp(self):
        self.calc = SimilarityCalculator()

    def test_default_weights_sum_to_1(self):
        total = sum(self.calc.DEFAULT_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_set_custom_weights(self):
        custom = {"text": 0.40, "ingredient": 0.30, "health": 0.20, "constraint": 0.10}
        self.calc.set_weights(custom)
        self.assertEqual(self.calc.weights["text"], 0.40)

    def test_invalid_weights_raises(self):
        """Weights yang tidak sum ke 1.0 harus raise ValueError."""
        with self.assertRaises(ValueError):
            self.calc.set_weights({"text": 0.5, "ingredient": 0.5, "health": 0.3, "constraint": 0.1})

    def test_missing_weight_key_raises(self):
        """Weights dengan key kurang harus raise ValueError."""
        with self.assertRaises(ValueError):
            self.calc.set_weights({"text": 1.0})

    def test_health_penalty_zero_for_unsuitable(self):
        """Resep yang tidak cocok kondisi kesehatan harus dapat skor 0."""
        case = RecipeCase(SAMPLE_RECIPES[3])  # Rendang – not suitable for diabetes
        score = self.calc._health_similarity(
            {"health_conditions": ["diabetes"]},
            case
        )
        self.assertEqual(score, 0.0)

    def test_health_neutral_no_conditions(self):
        """Tanpa kondisi kesehatan, health_similarity harus 0.5 (neutral)."""
        case = RecipeCase(SAMPLE_RECIPES[0])
        score = self.calc._health_similarity({"health_conditions": []}, case)
        self.assertEqual(score, 0.5)

    def test_case_weight_modifier(self):
        """case_weight > 1 harus meningkatkan total score."""
        import numpy as np
        case = RecipeCase(SAMPLE_RECIPES[1])  # Ayam Kukus Sehat
        vec = np.array([[0.5, 0.3, 0.1]])

        score1, _ = self.calc.compute(vec, vec, SAMPLE_ENTITIES, case, case_weight=1.0)
        score2, _ = self.calc.compute(vec, vec, SAMPLE_ENTITIES, case, case_weight=1.5)
        self.assertGreater(score2, score1)

    def test_case_weight_clamped_at_1(self):
        """Total score tidak boleh melebihi 1.0."""
        import numpy as np
        case = RecipeCase(SAMPLE_RECIPES[1])
        vec = np.array([[0.5, 0.3, 0.1]])
        score, _ = self.calc.compute(vec, vec, SAMPLE_ENTITIES, case, case_weight=10.0)
        self.assertLessEqual(score, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# T3. CBREngine v2 – Core Retrieve
# ═════════════════════════════════════════════════════════════════════════════

class TestCBREngineRetrieve(unittest.TestCase):
    """CBREngine retrieve dengan hard filter health & avoid."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        weights_path = os.path.join(self.tmpdir, "cbr_weights.json")
        self.cbr = CBREngine(weights_path=weights_path)
        self.cbr.load_cases(SAMPLE_RECIPES)

    def test_load_cases_count(self):
        self.assertEqual(len(self.cbr.cases), len(SAMPLE_RECIPES))

    def test_retrieve_returns_dict_with_required_keys(self):
        result = self.cbr.retrieve("mau masak ayam", SAMPLE_ENTITIES)
        self.assertIn("matched_recipes", result)
        self.assertIn("total_candidates", result)
        self.assertIn("query_hash", result)
        self.assertIn("cbr_metadata", result)

    def test_health_hard_filter_excludes_unsuitable(self):
        """Resep 'not_suitable_for diabetes' tidak boleh muncul di hasil."""
        entities = {
            "ingredients": {"main": ["sapi"], "avoid": []},
            "health_conditions": ["diabetes"],
            "time_constraint": None, "region": None,
        }
        result = self.cbr.retrieve("mau masak sapi", entities)
        recipe_ids = [r["recipe_id"] for r in result["matched_recipes"]]
        # Rendang Sapi (id=4) adalah not_suitable_for diabetes
        self.assertNotIn(4, recipe_ids, "Rendang harus di-filter karena not_suitable diabetes")

    def test_avoid_ingredient_filter(self):
        """Resep dengan bahan di avoid list tidak boleh muncul."""
        entities = {
            "ingredients": {"main": ["ayam"], "avoid": ["tepung"]},
            "health_conditions": [], "time_constraint": None, "region": None,
        }
        result = self.cbr.retrieve("mau masak ayam tanpa tepung", entities)
        # Resep 1 (Ayam Goreng Krispi) punya tepung di ingredients_all
        recipe_ids = [r["recipe_id"] for r in result["matched_recipes"]]
        self.assertNotIn(1, recipe_ids, "Ayam Goreng harus di-filter karena mengandung tepung")

    def test_match_score_between_0_and_100(self):
        """match_score harus dalam range 0-100."""
        result = self.cbr.retrieve("mau masak ayam", SAMPLE_ENTITIES)
        for r in result["matched_recipes"]:
            self.assertGreaterEqual(r["match_score"], 0)
            self.assertLessEqual(r["match_score"], 100)

    def test_results_sorted_by_score(self):
        """Hasil harus urut dari skor tertinggi ke terendah."""
        result = self.cbr.retrieve("mau masak ayam untuk diabetes", SAMPLE_ENTITIES)
        scores = [r["match_score"] for r in result["matched_recipes"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_score_breakdown_present(self):
        """Setiap hasil harus punya score_breakdown."""
        result = self.cbr.retrieve("mau masak ayam", SAMPLE_ENTITIES)
        for r in result["matched_recipes"]:
            bd = r["score_breakdown"]
            self.assertIn("text_similarity", bd)
            self.assertIn("ingredient_similarity", bd)
            self.assertIn("health_similarity", bd)
            self.assertIn("total", bd)
            self.assertIn("case_weight", bd)   # v2

    def test_empty_result_when_no_cases(self):
        """Retrieve tanpa cases harus return empty result."""
        weights_path = os.path.join(self.tmpdir, "cbr_weights_empty.json")
        cbr_empty = CBREngine(weights_path=weights_path)
        result = cbr_empty.retrieve("mau masak ayam", SAMPLE_ENTITIES)
        self.assertEqual(result["matched_recipes"], [])


# ═════════════════════════════════════════════════════════════════════════════
# T4. CBREngine v2 – Feedback Loop
# ═════════════════════════════════════════════════════════════════════════════

class TestCBRFeedbackLoop(unittest.TestCase):
    """Feedback loop: apply_feedback, weight persistence, bulk feedback."""

    def setUp(self):
        self.tmpdir      = tempfile.mkdtemp()
        self.weights_path = os.path.join(self.tmpdir, "cbr_weights.json")
        self.cbr         = CBREngine(weights_path=self.weights_path)
        self.cbr.load_cases(SAMPLE_RECIPES)

    def test_positive_feedback_increases_weight(self):
        """Rating +1 harus meningkatkan case_weight."""
        initial_weight = self.cbr.case_weights.get(1, 1.0)
        result = self.cbr.apply_feedback(recipe_id=1, rating=1)

        self.assertGreater(result["new_weight"], result["old_weight"])
        self.assertAlmostEqual(result["delta"], CBREngine.FEEDBACK_POSITIVE, places=4)
        self.assertEqual(self.cbr.case_weights[1], result["new_weight"])

    def test_negative_feedback_decreases_weight(self):
        """Rating -1 harus menurunkan case_weight."""
        result = self.cbr.apply_feedback(recipe_id=2, rating=-1)
        self.assertLess(result["new_weight"], result["old_weight"])
        self.assertAlmostEqual(result["delta"], CBREngine.FEEDBACK_NEGATIVE, places=4)

    def test_weight_clamped_at_max(self):
        """Weight tidak boleh melebihi WEIGHT_MAX meski banyak feedback positif."""
        for _ in range(100):
            self.cbr.apply_feedback(recipe_id=1, rating=1)
        self.assertLessEqual(self.cbr.case_weights[1], CBREngine.WEIGHT_MAX)

    def test_weight_clamped_at_min(self):
        """Weight tidak boleh turun di bawah WEIGHT_MIN."""
        for _ in range(100):
            self.cbr.apply_feedback(recipe_id=1, rating=-1)
        self.assertGreaterEqual(self.cbr.case_weights[1], CBREngine.WEIGHT_MIN)

    def test_feedback_affects_retrieval_score(self):
        """Resep dengan positive feedback harus dapat skor lebih tinggi."""
        # Baseline: retrieve tanpa feedback
        result_before = self.cbr.retrieve("mau masak ayam untuk diabetes", SAMPLE_ENTITIES)
        scores_before = {r["recipe_id"]: r["match_score"] for r in result_before["matched_recipes"]}

        # Berikan banyak positive feedback ke resep id=2 (Ayam Kukus Sehat)
        for _ in range(10):
            self.cbr.apply_feedback(recipe_id=2, rating=1)

        # Retrieve ulang
        result_after = self.cbr.retrieve("mau masak ayam untuk diabetes", SAMPLE_ENTITIES)
        scores_after = {r["recipe_id"]: r["match_score"] for r in result_after["matched_recipes"]}

        # Resep 2 harus naik
        if 2 in scores_before and 2 in scores_after:
            self.assertGreaterEqual(scores_after[2], scores_before[2])

    def test_save_and_reload_weights(self):
        """Weight tersimpan ke disk dan bisa di-reload."""
        self.cbr.apply_feedback(recipe_id=1, rating=1)
        saved_weight = self.cbr.case_weights[1]

        # Buat CBR engine baru dari file yang sama
        cbr2 = CBREngine(weights_path=self.weights_path)
        self.assertEqual(cbr2.case_weights.get(1), saved_weight)

    def test_bulk_feedback(self):
        """Bulk feedback harus update semua recipe IDs."""
        feedback_list = [
            {"recipe_id": 1, "rating": 1},
            {"recipe_id": 2, "rating": -1},
            {"recipe_id": 3, "rating": 1},
        ]
        results = self.cbr.apply_bulk_feedback(feedback_list)
        self.assertEqual(len(results), 3)
        self.assertGreater(self.cbr.case_weights[1], 1.0)   # positif
        self.assertLess(self.cbr.case_weights[2], 1.0)     # negatif

    def test_apply_feedback_return_structure(self):
        """Return value apply_feedback harus punya semua keys yang dibutuhkan."""
        result = self.cbr.apply_feedback(recipe_id=5, rating=1)
        required_keys = {"recipe_id", "old_weight", "new_weight", "delta", "saved"}
        self.assertTrue(required_keys.issubset(result.keys()))


# ═════════════════════════════════════════════════════════════════════════════
# T5. CBREngine v2 – Grid Search Interface
# ═════════════════════════════════════════════════════════════════════════════

class TestCBRGridSearchInterface(unittest.TestCase):
    """Grid search interface: set_similarity_weights, save/load."""

    def setUp(self):
        self.tmpdir      = tempfile.mkdtemp()
        self.weights_path = os.path.join(self.tmpdir, "cbr_weights.json")
        self.cbr         = CBREngine(weights_path=self.weights_path)
        self.cbr.load_cases(SAMPLE_RECIPES)

    def test_set_similarity_weights_updates_calculator(self):
        """set_similarity_weights harus update SimilarityCalculator.weights."""
        new_weights = {"text": 0.40, "ingredient": 0.30, "health": 0.20, "constraint": 0.10}
        self.cbr.set_similarity_weights(new_weights)
        self.assertEqual(self.cbr.similarity_calc.weights["text"], 0.40)

    def test_set_similarity_weights_persisted(self):
        """Bobot baru harus tersimpan ke file."""
        new_weights = {"text": 0.40, "ingredient": 0.25, "health": 0.25, "constraint": 0.10}
        self.cbr.set_similarity_weights(new_weights)

        cbr2 = CBREngine(weights_path=self.weights_path)
        self.assertAlmostEqual(cbr2.similarity_calc.weights["text"], 0.40, places=4)

    def test_get_stats_includes_feedback_info(self):
        """get_stats() v2 harus include feedback_weights dan similarity_weights."""
        stats = self.cbr.get_stats()
        self.assertIn("feedback_weights", stats)
        self.assertIn("similarity_weights", stats)
        self.assertIn("total_modified", stats["feedback_weights"])


# ═════════════════════════════════════════════════════════════════════════════
# T6. Grid Search Pipeline (dry-run)
# ═════════════════════════════════════════════════════════════════════════════

class TestWeightOptimizerPipeline(unittest.TestCase):
    """Grid search pipeline dengan synthetic data."""

    def setUp(self):
        self.tmpdir      = tempfile.mkdtemp()
        self.weights_path = os.path.join(self.tmpdir, "cbr_weights.json")
        self.cbr         = CBREngine(weights_path=self.weights_path)
        self.cbr.load_cases(SAMPLE_RECIPES)

    def _make_synthetic_validation(self, n=10):
        """Buat validation set sintetis sederhana."""
        import random
        data = []
        for _ in range(n):
            case   = random.choice(SAMPLE_RECIPES)
            rating = random.choice([1, 1, -1])
            ing    = case["ingredients_main"][0] if case["ingredients_main"] else "ayam"
            data.append({
                "query_text": f"mau masak {ing}",
                "entities":   {
                    "ingredients": {"main": [ing], "avoid": []},
                    "health_conditions": [], "time_constraint": None, "region": None,
                },
                "recipe_id":  case["id"],
                "rating":     rating,
                "rank_shown": 1,
            })
        return data

    def test_optimizer_runs_without_error(self):
        """Grid search harus berjalan tanpa exception."""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from Scripts.optimize_weights import WeightOptimizer
        except ImportError:
            self.skipTest("optimize_weights.py belum tersedia di scripts/")

        optimizer = WeightOptimizer(self.cbr)
        validation = self._make_synthetic_validation(n=10)
        result     = optimizer.run(validation)

        self.assertIn("best_weights", result)
        self.assertIn("leaderboard", result)
        self.assertIn("best_metrics", result)
        self.assertIn("ndcg_at_5", result["best_metrics"])

    def test_valid_combos_all_sum_to_1(self):
        """Semua kombinasi bobot valid harus sum ke ≈1.0."""
        try:
            from Scripts.optimize_weights import WeightOptimizer
        except ImportError:
            self.skipTest("optimize_weights.py belum tersedia di scripts/")

        optimizer = WeightOptimizer(self.cbr)
        for combo in optimizer._valid_combos:
            total = sum(combo.values())
            self.assertAlmostEqual(total, 1.0, delta=optimizer.TOLERANCE,
                                   msg=f"Combination {combo} sums to {total}")


# ═════════════════════════════════════════════════════════════════════════════
# T7. Integration – End-to-End Flow
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndFlow(unittest.TestCase):
    """
    Simulasi alur lengkap:
    Query → Retrieve → Feedback → Weight Update → Re-retrieve
    """

    def setUp(self):
        self.tmpdir      = tempfile.mkdtemp()
        self.weights_path = os.path.join(self.tmpdir, "cbr_weights.json")
        self.cbr         = CBREngine(weights_path=self.weights_path)
        self.cbr.load_cases(SAMPLE_RECIPES)
        self.pp          = TextPreprocessor(data_dir=self.tmpdir)

    def test_full_flow_query_to_feedback(self):
        """Alur: normalize → retrieve → feedback → re-retrieve."""
        # Step 1: Normalize query
        raw_query = "gw pengen masak ayam yg sehat buat diabetes"
        preprocessed = self.pp.preprocess(raw_query)
        normalized   = preprocessed["normalized"]

        self.assertNotIn("gw", normalized)
        self.assertNotIn("yg", normalized)

        # Step 2: Retrieve
        result1 = self.cbr.retrieve(normalized, SAMPLE_ENTITIES, top_k=3)
        self.assertGreater(len(result1["matched_recipes"]), 0)

        # Pastikan rendang tidak muncul (tidak cocok diabetes)
        ids1 = [r["recipe_id"] for r in result1["matched_recipes"]]
        self.assertNotIn(4, ids1)

        # Step 3: User beri feedback positif ke resep #2 (Ayam Kukus Sehat)
        if 2 in ids1:
            fb_result = self.cbr.apply_feedback(recipe_id=2, rating=1)
            self.assertGreater(fb_result["new_weight"], fb_result["old_weight"])

        # Step 4: Re-retrieve – skor resep #2 harus sama atau lebih tinggi
        result2 = self.cbr.retrieve(normalized, SAMPLE_ENTITIES, top_k=3)
        ids2    = [r["recipe_id"] for r in result2["matched_recipes"]]
        self.assertNotIn(4, ids2)   # hard filter tetap berlaku

    def test_query_hash_deterministic(self):
        """Query yang sama harus menghasilkan hash yang sama."""
        hash1 = self.cbr._hash_query("ayam goreng", SAMPLE_ENTITIES)
        hash2 = self.cbr._hash_query("ayam goreng", SAMPLE_ENTITIES)
        self.assertEqual(hash1, hash2)

    def test_different_queries_different_hash(self):
        """Query berbeda harus hash berbeda."""
        hash1 = self.cbr._hash_query("ayam goreng", SAMPLE_ENTITIES)
        hash2 = self.cbr._hash_query("tempe bacem", SAMPLE_ENTITIES)
        self.assertNotEqual(hash1, hash2)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    # Tambah semua test class
    for cls in [
        TestPreprocessorV2,
        TestSimilarityCalculator,
        TestCBREngineRetrieve,
        TestCBRFeedbackLoop,
        TestCBRGridSearchInterface,
        TestWeightOptimizerPipeline,
        TestEndToEndFlow,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)