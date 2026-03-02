#!/usr/bin/env python3
"""
scripts/optimize_weights.py
Grid search untuk optimasi bobot komponen similarity CBR.

Workflow:
    1. Load feedback historis dari DB (via export CSV/JSON dari Laravel)
    2. Jalankan grid search di semua kombinasi bobot
    3. Evaluasi dengan NDCG@5 dan Precision@3
    4. Simpan bobot terbaik ke models/cbr_weights.json
    5. (Opsional) Push update ke Flask via /api/cbr/weights

Usage:
    python scripts/optimize_weights.py --feedback data/feedback_export.json
    python scripts/optimize_weights.py --feedback data/feedback_export.json --push-to-flask
    python scripts/optimize_weights.py --dry-run  # pakai feedback sintetis untuk test

Requirement data feedback_export.json:
    [
        {
            "query_text":     "mau masak ayam untuk diabetes",
            "entities":       {...},
            "recipe_id":      15,
            "rating":         1,    // 1=positif, -1=negatif
            "rank_shown":     1     // posisi saat ditampilkan
        },
        ...
    ]
"""

import argparse
import json
import os
import sys
import itertools
import requests
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cbr_engine import CBREngine, SimilarityCalculator


# ── Metric Helpers ─────────────────────────────────────────────────────────────

def dcg_at_k(ratings: List[int], k: int) -> float:
    """Compute DCG@k."""
    ratings = ratings[:k]
    return sum(
        (2 ** rel - 1) / np.log2(i + 2)
        for i, rel in enumerate(ratings)
    )

def ndcg_at_k(ratings: List[int], ideal_ratings: List[int], k: int) -> float:
    """Compute NDCG@k."""
    ideal = sorted(ideal_ratings, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ratings, k) / ideal_dcg

def precision_at_k(ratings: List[int], k: int, threshold: int = 1) -> float:
    """Compute Precision@k (rating >= threshold dianggap relevan)."""
    relevant = sum(1 for r in ratings[:k] if r >= threshold)
    return relevant / k


# ── Grid Search ─────────────────────────────────────────────────────────────────

class WeightOptimizer:
    """
    Grid search sederhana untuk bobot similarity.

    Parameter search space bisa dikecilkan untuk efisiensi.
    Dengan 4 nilai per komponen dan 4 komponen → 4^4 = 256 kombinasi.
    Filter kombinasi yang totalnya != 1.0 akan mereduksi ke ~30-50 valid combinations.
    """

    SEARCH_SPACE = {
        "text":       [0.25, 0.30, 0.35, 0.40],
        "ingredient": [0.20, 0.25, 0.30, 0.35],
        "health":     [0.15, 0.20, 0.25, 0.30],
        "constraint": [0.05, 0.10, 0.15, 0.20],
    }

    NDCG_K      = 5
    PRECISION_K = 3
    TOLERANCE   = 0.01  # toleransi sum weight

    def __init__(self, cbr_engine: CBREngine):
        self.cbr = cbr_engine
        self._valid_combos: List[Dict] = []
        self._precompute_valid_combos()

    def _precompute_valid_combos(self):
        """Pre-filter kombinasi yang sum ≈ 1.0."""
        keys   = list(self.SEARCH_SPACE.keys())
        values = [self.SEARCH_SPACE[k] for k in keys]

        for combo in itertools.product(*values):
            total = sum(combo)
            if abs(total - 1.0) <= self.TOLERANCE:
                self._valid_combos.append(dict(zip(keys, combo)))

        print(f"  ✓ Valid weight combinations: {len(self._valid_combos)}")

    def evaluate(self, weights: Dict[str, float], validation_set: List[Dict]) -> Dict:
        """
        Evaluasi satu set weights di validation_set.

        validation_set: list of {query_text, entities, recipe_id, rating, rank_shown}
        """
        self.cbr.similarity_calc.set_weights(weights)

        ndcg_scores = []
        precision_scores = []

        for item in validation_set:
            query_text = item["query_text"]
            entities   = item["entities"]
            expected_recipe = item["recipe_id"]
            user_rating     = item["rating"]

            result = self.cbr.retrieve(query_text, entities, top_k=self.NDCG_K)
            matched_ids = [r["recipe_id"] for r in result["matched_recipes"]]

            # Construct rating vector berdasarkan apakah resep yang diberi feedback
            # muncul di top-k dan di posisi mana
            rating_vec = []
            for rid in matched_ids:
                if rid == expected_recipe:
                    rating_vec.append(max(0, user_rating))  # convert -1 ke 0
                else:
                    rating_vec.append(0)

            # Pad jika kurang dari k
            while len(rating_vec) < self.NDCG_K:
                rating_vec.append(0)

            ideal = sorted(rating_vec, reverse=True)
            ndcg_scores.append(ndcg_at_k(rating_vec, ideal, self.NDCG_K))
            precision_scores.append(precision_at_k(rating_vec, self.PRECISION_K))

        return {
            "ndcg_at_5":      round(np.mean(ndcg_scores), 4) if ndcg_scores else 0.0,
            "precision_at_3": round(np.mean(precision_scores), 4) if precision_scores else 0.0,
            "n_queries":      len(validation_set),
        }

    def run(self, validation_set: List[Dict]) -> Dict:
        """
        Jalankan grid search.
        Returns best weights + full leaderboard.
        """
        if len(validation_set) < 10:
            print(f"  ⚠ Validation set kecil ({len(validation_set)} sampel). "
                  f"Hasil mungkin tidak reliable. Disarankan minimal 50 sampel.")

        print(f"\n  Running grid search ({len(self._valid_combos)} combinations × "
              f"{len(validation_set)} validation samples)...")

        leaderboard = []
        for i, weights in enumerate(self._valid_combos):
            metrics = self.evaluate(weights, validation_set)
            leaderboard.append({**weights, **metrics})
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(self._valid_combos)} done...")

        # Sort by combined score (NDCG lebih penting)
        leaderboard.sort(
            key=lambda x: 0.7 * x["ndcg_at_5"] + 0.3 * x["precision_at_3"],
            reverse=True
        )

        best = leaderboard[0]
        best_weights = {
            k: best[k] for k in ("text", "ingredient", "health", "constraint")
        }

        print(f"\n  ✅ Best weights: {best_weights}")
        print(f"     NDCG@5={best['ndcg_at_5']:.4f} | Precision@3={best['precision_at_3']:.4f}")
        print(f"\n  Previous weights: {SimilarityCalculator.DEFAULT_WEIGHTS}")

        return {
            "best_weights":  best_weights,
            "best_metrics":  {"ndcg_at_5": best["ndcg_at_5"], "precision_at_3": best["precision_at_3"]},
            "leaderboard":   leaderboard[:10],    # top 10
            "n_combinations": len(self._valid_combos),
        }


# ── Feedback Loader ─────────────────────────────────────────────────────────────

def load_feedback(path: str) -> List[Dict]:
    """Load feedback dari JSON export Laravel."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Filter hanya feedback dengan rating valid
    valid = [
        item for item in data
        if item.get("rating") in (1, -1)
        and item.get("recipe_id")
        and item.get("query_text")
    ]
    print(f"  ✓ Loaded {len(valid)}/{len(data)} valid feedback samples")
    return valid


def make_synthetic_feedback(cbr: CBREngine, n: int = 50) -> List[Dict]:
    """
    Buat feedback sintetis untuk dry-run (saat belum ada data riil).
    Hanya untuk test pipeline, bukan untuk production tuning!
    """
    if not cbr.cases:
        return []

    import random
    feedback = []
    for _ in range(n):
        case    = random.choice(cbr.cases)
        rating  = random.choice([1, 1, -1])  # bias positif
        ing     = random.choice(case.ingredients_main) if case.ingredients_main else "ayam"
        query   = f"mau masak {ing}"
        entities = {
            "ingredients": {"main": [ing], "avoid": []},
            "health_conditions": [],
            "time_constraint": None,
            "region": None,
        }
        feedback.append({
            "query_text": query,
            "entities":   entities,
            "recipe_id":  case.recipe_id,
            "rating":     rating,
            "rank_shown": 1,
        })
    return feedback


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CBR Weight Optimizer – Grid Search")
    parser.add_argument("--feedback",       help="Path ke feedback_export.json")
    parser.add_argument("--cbr-index",      default="data/cbr_index_sample.json",
                        help="Path ke sample recipe data untuk index")
    parser.add_argument("--output",         default="models/cbr_weights.json",
                        help="Path output weights")
    parser.add_argument("--push-to-flask",  action="store_true",
                        help="Push best weights ke Flask via API setelah selesai")
    parser.add_argument("--flask-url",      default="http://127.0.0.1:5000")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Gunakan feedback sintetis (untuk test pipeline)")
    args = parser.parse_args()

    print("=" * 65)
    print("  CBR WEIGHT OPTIMIZER – GRID SEARCH")
    print("=" * 65)

    # 1. Init CBR Engine
    cbr = CBREngine(weights_path=args.output)

    # 2. Load CBR Index (butuh data resep untuk run retrieve)
    if os.path.isfile(args.cbr_index):
        with open(args.cbr_index, encoding="utf-8") as f:
            recipes = json.load(f)
        cbr.load_cases(recipes)
        print(f"  ✓ Loaded {len(cbr.cases)} recipe cases")
    else:
        print(f"  ⚠ {args.cbr_index} tidak ditemukan.")
        print("    Untuk grid search, CBR engine butuh data resep.")
        print("    Export data dengan: php artisan cbr:export-index > data/cbr_index_sample.json")
        if not args.dry_run:
            sys.exit(1)

    # 3. Load feedback
    if args.dry_run:
        print("\n  [DRY-RUN] Menggunakan feedback sintetis...")
        validation_set = make_synthetic_feedback(cbr, n=30)
    elif args.feedback and os.path.isfile(args.feedback):
        validation_set = load_feedback(args.feedback)
    else:
        print(f"  ✗ Feedback file tidak ditemukan: {args.feedback}")
        print("    Export feedback dengan: php artisan feedback:export > data/feedback_export.json")
        sys.exit(1)

    if not validation_set:
        print("  ✗ Tidak ada data validasi. Abort.")
        sys.exit(1)

    # 4. Run grid search
    optimizer = WeightOptimizer(cbr)
    result    = optimizer.run(validation_set)

    # 5. Simpan best weights
    best_weights = result["best_weights"]
    cbr.set_similarity_weights(best_weights)
    print(f"\n  ✅ Weights disimpan ke {args.output}")

    # 6. Save leaderboard report
    report_path = "data/grid_search_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Leaderboard disimpan ke {report_path}")

    # 7. (Opsional) Push ke Flask
    if args.push_to_flask:
        print(f"\n  Pushing weights ke Flask ({args.flask_url})...")
        try:
            resp = requests.post(
                f"{args.flask_url}/api/cbr/weights",
                json={"weights": best_weights},
                headers={"X-Internal-Key": os.getenv("NLP_SERVICE_KEY", "")},
                timeout=10,
            )
            if resp.status_code == 200:
                print("  ✅ Weights berhasil di-push ke Flask service")
            else:
                print(f"  ⚠ Flask response: {resp.status_code} – {resp.text[:100]}")
        except Exception as e:
            print(f"  ⚠ Gagal push ke Flask: {e}")

    print("\n" + "=" * 65)
    print("  SELESAI")
    print("=" * 65)


if __name__ == "__main__":
    main()