# api/flask_api.py  (v2 – CBR + Redis + Feedback Loop)
# Flask NLP Service – kala_rasa_jtv
#
# Arsitektur:
#   POST /api/chat                → NLP processing + slot-filling
#   POST /api/cbr/index           → Laravel mengirim data resep → bangun CBR index
#   POST /api/cbr/match           → CBR similarity matching
#   POST /api/cbr/popular         → Rekomendasi populer (cached)
#   POST /api/feedback            → Feedback user (👍/👎)
#   POST /api/cbr/weights         → Update bobot similarity (Grid Search)
#   POST /api/reload-dicts        → Hot-reload kamus NLP tanpa restart
#   GET  /api/session/<id>/context
#   DELETE /api/session/<id>
#   GET  /health

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conversational_ai import ConversationalAI
from src.cbr_engine import CBREngine
from src.chache_manager import CacheManager

app = Flask(__name__)
CORS(app)

NLP_SERVICE_KEY = os.getenv("NLP_SERVICE_KEY", "")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# ── Inisialisasi services ─────────────────────────────────────────────────────

print("Initializing services...")

try:
    cache = CacheManager()
    print("✓ Cache manager ready")
except Exception as e:
    print(f"✗ Cache manager failed: {e}")
    cache = None

try:
    ai = ConversationalAI(model_dir=MODEL_DIR)
    print("✓ ConversationalAI (NLP) ready")
except Exception as e:
    print(f"✗ NLP init failed: {e}")
    ai = None

try:
    cbr = CBREngine(weights_path=os.getenv("CBR_WEIGHTS_PATH", "models/cbr_weights.json"))
    print("✓ CBR Engine ready")
except Exception as e:
    print(f"✗ CBR init failed: {e}")
    cbr = None

print("─" * 50)


# ── Auth helper ───────────────────────────────────────────────────────────────

def _auth_ok() -> bool:
    if not NLP_SERVICE_KEY:
        return True
    return request.headers.get("X-Internal-Key", "") == NLP_SERVICE_KEY


# ── Conversation state helper ─────────────────────────────────────────────────

def _conversation_state(nlp: Dict, ctx: Dict) -> str:
    intent = nlp.get("intent", "unknown")
    status = nlp.get("status", "fallback")
    action = nlp.get("action", "")

    non_recipe = {"chitchat","lihat_detail"}
    if intent in non_recipe:
        return "done"
    if status == "clarification" or action == "ask_clarification":
        return "clarifying"
    if status == "fallback":
        return "collecting"

    col = ctx.get("collected_entities", {})
    if (col.get("ingredients", {}).get("main")
            or col.get("health_conditions")
            or col.get("region")
            or col.get("time_constraint") is not None):
        return "ready"
    return "collecting"


def _extract_recipe_index(message: str, nlp: Dict) -> Optional[int]:
    if nlp.get("intent") != "lihat_detail":
        return None
    m = re.search(r"\b(\d+)\b", message)
    if m:
        return int(m.group(1))
    ordinals = {
        "pertama": 1, "satu": 1, "kedua": 2, "dua": 2,
        "ketiga": 3, "tiga": 3, "keempat": 4, "empat": 4,
        "kelima": 5, "lima": 5,
    }
    msg_lower = message.lower()
    for word, idx in ordinals.items():
        if word in msg_lower:
            return idx
    return 1


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1: /api/chat  (NLP + slot-filling, existing flow)
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Endpoint utama percakapan.
    Laravel mengirim session_id, user_id, message.
    Flask memproses NLP, slot-filling, dan mengembalikan intent + entities + response.

    Flow baru:
    - Jika conversation_state == 'ready', Laravel harus memanggil /api/cbr/match
      dengan entities yang dikembalikan untuk mendapatkan rekomendasi resep.
    """
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    if not ai:
        return jsonify({"success": False, "error": "NLP service unavailable"}), 503

    data       = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip()
    user_id    = data.get("user_id", "").strip()
    message    = data.get("message", "").strip()

    if not session_id:
        return jsonify({"success": False, "error": "'session_id' required"}), 400
    if not user_id:
        return jsonify({"success": False, "error": "'user_id' required"}), 400
    if not message:
        return jsonify({"success": False, "error": "'message' required"}), 400

    try:
        if data.get("reset"):
            ai.reset_context(session_id)
            if cache:
                cache.delete_session(session_id)
                # BUG FIX: Hapus juga similarity cache agar query lama
                # tidak ter-serve ke session baru yang memakai session_id sama.
                cache.invalidate_cbr_cache()

        result = ai.process_message(session_id, message, reset=bool(data.get("reset")))
        nlp    = result["nlp_result"]
        ctx    = result["context"]

        # Persist session context ke Redis
        if cache:
            cache.set_session(session_id, ctx)

        conv_state   = _conversation_state(nlp, ctx)
        recipe_index = _extract_recipe_index(message, nlp)

        # Cache entity extraction result
        if cache and nlp.get("entities"):
            query_hash = _make_entity_hash(message)
            cache.set_entity_cache(query_hash, nlp["entities"])

        return jsonify({
            "success": True,

            # NLP result
            "intent":     nlp["intent"],
            "confidence": nlp["confidence"],
            "status":     nlp["status"],
            "entities":   nlp["entities"],

            # Accumulated context
            "context_entities":   ctx["collected_entities"],

            # Action signals for Laravel
            "action":             nlp["action"],
            "conversation_state": conv_state,

            # Response
            "bot_message":            result["response"],
            "quick_replies":          result.get("suggestions", []),
            "clarification_needed":   nlp["status"] == "clarification",
            "clarification_question": (
                nlp["message"] if nlp["status"] == "clarification" else None
            ),

            "recipe_index": recipe_index,
            "turn_count":   ctx["conversation_turns"],
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2: /api/cbr/index  – Laravel mengirim data resep untuk di-index
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/cbr/index", methods=["POST"])
def cbr_index():
    """
    Laravel mengirim semua data resep aktif untuk membangun CBR index.

    Dipanggil:
      - Saat Flask service start (via Laravel artisan command)
      - Saat ada resep baru di-approve
      - Scheduled setiap X menit (opsional)

    Request body:
    {
        "recipes": [
            {
                "id": 1,
                "nama": "Ayam Goreng",
                "waktu_masak": 30,
                "region": "Jawa",
                "kategori": "goreng",
                "deskripsi": "...",
                "avg_rating": 4.5,
                "view_count": 100,
                "ingredients_main": ["ayam"],
                "ingredients_all": ["ayam", "tepung", "telur"],
                "suitable_for": ["diabetes"],
                "not_suitable_for": ["kolesterol"]
            }
        ]
    }
    """
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    if not cbr:
        return jsonify({"success": False, "error": "CBR engine unavailable"}), 503

    data    = request.get_json(silent=True) or {}
    recipes = data.get("recipes", [])

    if not recipes:
        return jsonify({"success": False, "error": "No recipes provided"}), 400

    try:
        count = cbr.load_cases(recipes)

        # Update cache dengan hash baru
        if cache:
            cache.set_index_hash(cbr._cases_hash)
            cache.invalidate_cbr_cache()  # invalidasi similarity cache lama

        return jsonify({
            "success":       True,
            "cases_indexed": count,
            "index_hash":    cbr._cases_hash,
            "stats":         cbr.get_stats(),
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3: /api/cbr/match  – CBR similarity matching
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/cbr/match", methods=["POST"])
def cbr_match():
    """
    Melakukan CBR similarity matching.

    Dipanggil oleh Laravel setelah conversation_state == 'ready'
    atau langsung dari ChatbotController.

    Request body:
    {
        "session_id": "abc123",
        "user_id": "42",
        "query_text": "mau masak ayam untuk diabetes",
        "entities": {
            "ingredients": {"main": ["ayam"], "avoid": []},
            "health_conditions": ["diabetes"],
            "time_constraint": 30,
            "region": null
        },
        "top_k": 5   // optional, default 5
    }

    Response:
    {
        "success": true,
        "matched_recipes": [
            {
                "rank_position": 1,
                "recipe_id": 3,
                "nama": "Ayam Kukus Sehat",
                "match_score": 87.50,        // → matched_recipes.match_score
                "waktu_masak": 25,
                "region": "jawa",
                "avg_rating": 4.3,
                "ingredients_main": ["ayam"],
                "suitable_for": ["diabetes"],
                "score_breakdown": {...}
            }
        ],
        "total_candidates": 12,
        "from_cache": false,
        "query_hash": "a1b2c3d4"            // untuk matched_recipes logging
    }
    """
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    if not cbr:
        return jsonify({"success": False, "error": "CBR engine unavailable"}), 503
    if not cbr.cases:
        return jsonify({"success": False, "error": "CBR index not built. Call /api/cbr/index first."}), 503

    data       = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    query_text = data.get("query_text", "").strip()
    entities   = data.get("entities", {})
    top_k      = min(int(data.get("top_k", 5)), 10)

    if not query_text and not entities:
        return jsonify({"success": False, "error": "query_text or entities required"}), 400

    try:
        # ── Check Redis cache ──────────────────────────────────────────
        query_hash = cbr._hash_query(query_text, entities)
        cached     = cache.get_similarity_cache(query_hash) if cache else None

        if cached:
            cached["from_cache"] = True
            # Refresh session TTL
            if cache and session_id:
                cache.refresh_session_ttl(session_id)
            return jsonify({"success": True, **cached})

        # ── CBR Retrieve ───────────────────────────────────────────────
        result = cbr.retrieve(query_text, entities, top_k=top_k)

        # ── Cache result ───────────────────────────────────────────────
        if cache and result["matched_recipes"]:
            cache.set_similarity_cache(query_hash, result)

        return jsonify({
            "success":    True,
            "from_cache": False,
            **result,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4: /api/cbr/popular  – Rekomendasi populer (cached)
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/cbr/popular", methods=["POST"])
def cbr_popular():
    """
    Laravel mengirim daftar resep populer untuk di-cache.
    Dipakai sebagai fallback saat tidak ada entities / query baru.

    Request body: { "recipes": [...] }  (format sama seperti /api/cbr/index)
    """
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    data    = request.get_json(silent=True) or {}
    recipes = data.get("recipes", [])

    # Cek cache dulu
    if not recipes and cache:
        cached = cache.get_popular_recipes()
        if cached:
            return jsonify({"success": True, "from_cache": True, **cached})

    if not recipes:
        return jsonify({"success": False, "error": "No recipes provided"}), 400

    payload = {"recipes": recipes, "count": len(recipes)}
    if cache:
        cache.set_popular_recipes(payload)

    return jsonify({"success": True, "from_cache": False, **payload})


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 5: /api/feedback  – Feedback user (👍/👎)
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """
    Terima feedback user dan update CBR case weight.

    Dipanggil dari Laravel ChatbotController setelah user klik 👍/👎.

    Request body:
    {
        "session_id":   "user_42_abc123",
        "user_id":      "42",
        "recipe_id":    15,
        "rating":       1,           // 1=positif, -1=negatif
        "feedback_type": "explicit", // "explicit" | "implicit"
        "query_hash":   "a1b2c3d4"  // opsional – untuk log
    }

    Response:
    {
        "success": true,
        "recipe_id":   15,
        "old_weight":  1.0,
        "new_weight":  1.08,
        "delta":       0.08
    }
    """
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    if not cbr:
        return jsonify({"success": False, "error": "CBR engine unavailable"}), 503

    data          = request.get_json(silent=True) or {}
    recipe_id     = data.get("recipe_id")
    rating        = data.get("rating")
    feedback_type = data.get("feedback_type", "explicit")

    if recipe_id is None:
        return jsonify({"success": False, "error": "'recipe_id' required"}), 400
    if rating not in (1, -1):
        return jsonify({"success": False, "error": "'rating' harus 1 atau -1"}), 400

    try:
        result = cbr.apply_feedback(int(recipe_id), int(rating))

        # Invalidate similarity cache – skor akan berubah setelah weight update
        if cache:
            cache.invalidate_cbr_cache()

        return jsonify({"success": True, **result})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 6: /api/cbr/weights  – Update bobot similarity (Grid Search)
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/cbr/weights", methods=["POST"])
def update_cbr_weights():
    """
    Update bobot komponen similarity dari hasil grid search.
    Hanya boleh dipanggil dari internal (misalnya setelah scripts/optimize_weights.py selesai).

    Request body:
    {
        "weights": {
            "text":       0.35,
            "ingredient": 0.30,
            "health":     0.20,
            "constraint": 0.15
        }
    }

    Response: { "success": true, "weights": {...}, "saved": true }
    """
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    if not cbr:
        return jsonify({"success": False, "error": "CBR engine unavailable"}), 503

    data    = request.get_json(silent=True) or {}
    weights = data.get("weights", {})

    required_keys = {"text", "ingredient", "health", "constraint"}
    if not required_keys.issubset(weights.keys()):
        return jsonify({
            "success": False,
            "error":   f"weights harus punya keys: {required_keys}"
        }), 400

    total = sum(weights.values())
    if not (0.99 <= total <= 1.01):
        return jsonify({
            "success": False,
            "error":   f"Jumlah weights harus ~1.0, dapat {total:.4f}"
        }), 400

    try:
        cbr.set_similarity_weights(weights)

        # Invalidate cache karena skor akan berubah
        if cache:
            cache.invalidate_cbr_cache()

        return jsonify({
            "success": True,
            "weights": cbr.similarity_calc.weights,
            "saved":   True,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 7: /api/reload-dicts  – Hot-reload kamus NLP tanpa restart
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/reload-dicts", methods=["POST"])
def reload_dictionaries():
    """
    Reload kamus NLP (informal_map.json, synonyms/*.json) tanpa restart service.
    Dipanggil setelah tim konten update file JSON kamus.

    Request body: {} (kosong)
    Response: { "success": true, "message": "Dictionaries reloaded" }
    """
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    if not ai:
        return jsonify({"success": False, "error": "NLP service unavailable"}), 503

    try:
        ai.nlp_engine.preprocessor.reload_dictionaries()
        return jsonify({
            "success": True,
            "message": "Dictionaries reloaded successfully",
            "stats": {
                "informal_map_size":   len(ai.nlp_engine.preprocessor.informal_map),
                "synonym_count":       len(ai.nlp_engine.preprocessor.synonym_map),
                "reverse_synonym_count": len(ai.nlp_engine.preprocessor._reverse_synonym),
            }
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 11: /api/nlp/retrain  – Retrain intent classifier dari conversation history
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/nlp/retrain", methods=["POST"])
def retrain_intent_classifier():
    """
    Retrain intent classifier menggunakan data conversation history dari Laravel.

    Dipanggil dari artisan command atau scheduler setelah cukup data terkumpul.
    
    Workflow feedback loop:
    1. Laravel artisan export data user_queries (status=ok, confidence>=0.75)
    2. POST ke endpoint ini dengan data tersebut
    3. Flask retrain model dengan data baru + built-in dataset
    4. Model disimpan ke disk, langsung aktif untuk request berikutnya

    Request body:
    {
        "history": [
            {
                "query_text": "saya ingin makan kambing",
                "intent": "cari_resep",
                "confidence": 0.82
            },
            ...
        ],
        "min_confidence": 0.75   // opsional, default 0.75
    }

    Response:
    {
        "success": true,
        "train_score": 0.97,
        "test_score": 0.89,
        "new_samples": 43
    }
    """
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    if not ai:
        return jsonify({"success": False, "error": "NLP service unavailable"}), 503

    data = request.get_json(silent=True) or {}
    history = data.get("history", [])
    min_confidence = float(data.get("min_confidence", 0.75))

    if not history:
        return jsonify({"success": False, "error": "history array is required"}), 400

    try:
        result = ai.nlp_engine.intent_classifier.retrain_from_conversation_history(
            history=history,
            model_dir=MODEL_DIR,
            min_confidence=min_confidence,
        )
        return jsonify({"success": True, **result})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 12: /api/cbr/rebuild  – Rebuild CBR index dengan force-refresh
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/cbr/rebuild", methods=["POST"])
def cbr_rebuild():
    """
    Force rebuild CBR index dari data terbaru (tanpa hash check).
    Berbeda dengan /api/cbr/index yang skip rebuild jika hash sama.

    Gunakan endpoint ini ketika:
    - Resep baru di-approve di database
    - Ada perubahan data resep (update bahan, suitability, dll.)
    - Setelah optimize_weights.py mengubah similarity weights
    - Saat maintenance / deployment baru

    Request body:
    {
        "recipes": [...],   // data resep terbaru dari Laravel
        "reason": "new_recipe_approved"  // opsional, untuk logging
    }

    Response:
    {
        "success": true,
        "cases_indexed": 87,
        "index_hash": "abc123",
        "reason": "new_recipe_approved",
        "cache_invalidated": true
    }
    """
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    if not cbr:
        return jsonify({"success": False, "error": "CBR engine unavailable"}), 503

    data = request.get_json(silent=True) or {}
    recipes = data.get("recipes", [])
    reason = data.get("reason", "manual_rebuild")

    if not recipes:
        return jsonify({"success": False, "error": "No recipes provided"}), 400

    try:
        # Force rebuild dengan cara reset hash dulu
        cbr._cases_hash = ""  # paksa rebuild meski data sama
        count = cbr.load_cases(recipes)

        # Invalidate ALL cache karena index baru
        if cache:
            cache.set_index_hash(cbr._cases_hash)
            cache.invalidate_cbr_cache()

        return jsonify({
            "success": True,
            "cases_indexed": count,
            "index_hash": cbr._cases_hash,
            "reason": reason,
            "cache_invalidated": True,
            "stats": cbr.get_stats(),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 13: /api/cbr/feedback/bulk  – Batch feedback untuk update case weights
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/cbr/feedback/bulk", methods=["POST"])
def cbr_feedback_bulk():
    """
    Batch update CBR case weights dari multiple feedback sekaligus.
    Dipakai untuk initial seeding dari historical feedback data.

    Request body:
    {
        "feedbacks": [
            {"recipe_id": 15, "rating": 1},
            {"recipe_id": 22, "rating": -1},
            ...
        ]
    }
    """
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    if not cbr:
        return jsonify({"success": False, "error": "CBR engine unavailable"}), 503

    data = request.get_json(silent=True) or {}
    feedbacks = data.get("feedbacks", [])

    if not feedbacks:
        return jsonify({"success": False, "error": "feedbacks array required"}), 400

    try:
        results = cbr.apply_bulk_feedback(feedbacks)

        # Invalidate cache setelah bulk update
        if cache:
            cache.invalidate_cbr_cache()

        return jsonify({
            "success": True,
            "updated": len(results),
            "results": results,
        })

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 8: /api/session/<id>/context  – Ambil konteks session
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/session/<session_id>/context", methods=["GET"])
def get_context(session_id: str):
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    # Coba ambil dari Redis dulu
    if cache:
        ctx_cached = cache.get_session(session_id)
        if ctx_cached:
            cache.refresh_session_ttl(session_id)
            return jsonify({
                "success":          True,
                "session_id":       session_id,
                "source":           "cache",
                "context_entities": ctx_cached.get("collected_entities", {}),
                "turn_count":       ctx_cached.get("conversation_turns", 0),
                "current_intent":   ctx_cached.get("current_intent"),
            })

    # Fallback ke in-memory
    ctx = ai.conversations.get(session_id) if ai else None
    if not ctx:
        return jsonify({"success": False, "error": "Session not found"}), 404

    return jsonify({
        "success":          True,
        "session_id":       session_id,
        "source":           "memory",
        "context_entities": ctx.collected_entities,
        "turn_count":       len(ctx.history),
        "current_intent":   ctx.current_intent,
    })


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 9: /api/session/<id>  – Hapus session
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/session/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    if not _auth_ok():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    deleted = False
    if ai and session_id in ai.conversations:
        ai.reset_context(session_id)
        deleted = True
    if cache:
        # BUG FIX: Gunakan invalidate_session_cache untuk membersihkan
        # SEMUA cache yang terkait session (session context + CBR similarity).
        # Sebelumnya hanya delete_session yang dipanggil, meninggalkan
        # similarity cache yang masih bisa di-hit oleh query baru.
        cache.invalidate_session_cache(session_id)
        deleted = True

    if deleted:
        return jsonify({"success": True, "message": "Session cleared"})
    return jsonify({"success": False, "error": "Session not found"}), 404


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT 10: /health  – Health check
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "service":         "kala_rasa_nlp_v2",
        "timestamp":       datetime.now().isoformat(),
        "nlp_ready":       ai is not None,
        "cbr_ready":       cbr is not None and len(cbr.cases) > 0,
        "cache_backend":   cache.get_stats()["backend"] if cache else "unavailable",
        "active_sessions": len(ai.conversations) if ai else 0,
        "cbr_stats":       cbr.get_stats() if cbr else {},
    })


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_entity_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)