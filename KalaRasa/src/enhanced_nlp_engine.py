# src/enhanced_nlp_engine.py
# Enhanced NLP Engine – diselaraskan dengan skema database kala_rasa_jtv
#
# Output JSON disesuaikan dengan kolom user_queries:
#   status     : enum('ok','fallback','clarification')
#   intent     : varchar(50) – sesuai daftar intent yang valid
#   confidence : decimal(3,2)
#   entities   : JSON sesuai struktur database (ingredients, health_conditions, dll.)

import os
import re
import json
from typing import Dict, List, Optional

from src.preprocessor import TextPreprocessor
from src.intent_classifier import IntentClassifier
from src.ner_extractor import NERExtractor


class EnhancedNLPEngine:
    """
    NLP Engine utama yang menggabungkan:
    - TextPreprocessor  (normalisasi kata informal)
    - IntentClassifier  (TF-IDF + Random Forest)
    - NERExtractor      (rule-based entity extraction, selaras DB)
    """

    # ── Status (sesuai user_queries.status ENUM) ─────────────────────
    STATUS_OK = "ok"
    STATUS_FALLBACK = "fallback"
    STATUS_CLARIFICATION = "clarification"

    # ── Actions (internal, tidak disimpan ke DB) ──────────────────────
    ACTION_MATCH_RECIPE = "match_recipe"
    ACTION_ASK_CLARIFICATION = "ask_clarification"
    ACTION_REJECT_INPUT = "reject_input"
    ACTION_SHOW_DETAIL = "show_detail"
    ACTION_CHITCHAT = "chitchat"
    ACTION_SHOW_RESTRICTIONS = "show_restrictions"

    # ── Threshold ─────────────────────────────────────────────────────
    MIN_CONFIDENCE = 0.35

    # ── Intent yang memerlukan slot bahan ─────────────────────────────
    RECIPE_INTENTS = {"cari_resep", "cari_resep_sehat", "filter_bahan"}

    # ── Map intent → action ───────────────────────────────────────────
    INTENT_ACTION_MAP = {
        "cari_resep": ACTION_MATCH_RECIPE,
        "cari_resep_sehat": ACTION_MATCH_RECIPE,
        "filter_bahan": ACTION_MATCH_RECIPE,
        "filter_waktu": ACTION_MATCH_RECIPE,
        "filter_region": ACTION_MATCH_RECIPE,
        "lihat_detail": ACTION_SHOW_DETAIL,
        "tanya_pantangan": ACTION_SHOW_RESTRICTIONS,
        "chitchat": ACTION_CHITCHAT,
    }

    # ── Kata kunci sederhana untuk fast-track ─────────────────────────
    SIMPLE_INGREDIENT_KEYWORDS = {
        "ayam", "ikan", "sapi", "kambing", "udang", "cumi", "kepiting",
        "tempe", "tahu", "telur", "sayur", "wortel", "bayam", "kangkung",
        "nasi", "mie", "pasta", "kentang", "singkong",
    }

    SIMPLE_INTENT_KEYWORDS = {
        "masak": "cari_resep", "bikin": "cari_resep", "buat": "cari_resep",
        "resep": "cari_resep", "makan": "cari_resep", "detail": "lihat_detail",
        "pantangan": "tanya_pantangan",
    }

    # ── Kosakata dikenal untuk gibberish detection ────────────────────
    KNOWN_VOCAB = {
        "saya", "aku", "gw", "mau", "ingin", "pengen", "ada", "tidak", "ga",
        "gak", "bisa", "boleh", "harus", "masak", "bikin", "buat", "goreng",
        "rebus", "panggang", "tumis", "ayam", "ikan", "sapi", "udang", "sayur",
        "tempe", "tahu", "nasi", "mie", "pasta", "telur", "daging", "resep",
        "masakan", "bahan", "bumbu", "cepat", "mudah", "simple", "diabetes",
        "kolesterol", "diet", "sehat", "alergi", "pedas", "manis", "asin",
        "gurih", "segar", "yang", "dengan", "untuk", "tanpa", "aja", "dong",
        "sih", "nih", "ya", "yuk", "ok", "oke", "halo", "hai", "terima",
        "kasih", "tolong", "bantu", "carikan", "cari", "tampilkan", "lihat",
        "detail", "simpan", "hapus", "kalori", "gizi",
        "pantangan", "hindari", "alergi", "region", "daerah",
    }

    def __init__(self, model_dir: str = "models"):
        """
        Inisialisasi engine.
        Jika model belum ada, otomatis training dari dataset built-in.
        """
        print("Initializing Enhanced NLP Engine...")

        self.preprocessor = TextPreprocessor(data_dir=os.getenv("NLP_DATA_DIR", "data"))
        print("  ✓ Preprocessor ready")

        self.intent_classifier = IntentClassifier()
        try:
            self.intent_classifier.load_model(model_dir)
            print("  ✓ Intent classifier loaded from disk")
        except (FileNotFoundError, OSError):
            print("  ℹ Model not found → training from built-in dataset...")
            self.intent_classifier.train_from_builtin()
            self.intent_classifier.save_model(model_dir)
            print("  ✓ Intent classifier trained & saved")

        self.ner_extractor = NERExtractor()
        print("  ✓ NER extractor ready")
        print("✓ Enhanced NLP Engine initialized\n")

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────

    def process(self, user_input: str) -> Dict:
        """
        Proses input user dan kembalikan hasil NLP.

        Output disesuaikan dengan kolom user_queries:
        {
            "status": "ok" | "fallback" | "clarification",
            "intent": str,                  # user_queries.intent
            "confidence": float,            # user_queries.confidence
            "entities": {                   # user_queries.entities (JSON)
                "ingredients": {
                    "main": [...],          # → ingredients.nama
                    "avoid": [...]          # → health_condition_restrictions
                },
                "cooking_methods": [...],
                "health_conditions": [...], # → health_conditions.nama
                "taste_preferences": [...],
                "time_constraint": int|None,# → recipes.waktu_masak
                "region": str|None          # → recipes.region
            },
            "action": str,                  # petunjuk untuk layer berikutnya
            "message": str                  # pesan untuk user
        }
        """
        try:
            # ── 0. Input kosong ───────────────────────────────────────
            if not user_input or not user_input.strip():
                return self._response(
                    status=self.STATUS_FALLBACK,
                    intent="unknown",
                    confidence=0.0,
                    entities={},
                    action=self.ACTION_ASK_CLARIFICATION,
                    message="Silakan tanyakan sesuatu tentang resep masakan 😊\n"
                            "Contoh: 'mau masak ayam goreng' atau 'resep untuk diabetes'",
                )

            text = user_input.strip()
            text_lower = text.lower()

            # ── 1. Fast-track: input sangat pendek / kata kunci tunggal ──
            fast = self._fast_track(text_lower)
            if fast:
                return fast

            # ── 2. Gibberish detection ────────────────────────────────
            if self._is_gibberish(text_lower):
                return self._response(
                    status=self.STATUS_FALLBACK,
                    intent="unknown",
                    confidence=0.0,
                    entities={},
                    action=self.ACTION_REJECT_INPUT,
                    message="Aku belum bisa memahami pesan kamu 😅\n"
                            "Coba tulis lebih jelas, misalnya:\n"
                            "• 'mau masak ayam goreng'\n"
                            "• 'resep untuk penderita diabetes'\n"
                            "• 'masakan padang yang tidak pedas'",
                )

            # ── 3. Preprocessing ──────────────────────────────────────
            preprocessed = self.preprocessor.preprocess(text)
            normalized = preprocessed["normalized"]

            # ── 4. Intent classification ──────────────────────────────
            intent_result = self.intent_classifier.predict(normalized)
            intent = intent_result["primary"]
            confidence = intent_result["confidence"]

            # ── 5. Low confidence → fallback ──────────────────────────
            if confidence < self.MIN_CONFIDENCE:
                return self._response(
                    status=self.STATUS_FALLBACK,
                    intent="unknown",
                    confidence=confidence,
                    entities={},
                    action=self.ACTION_ASK_CLARIFICATION,
                    message="Maaf, aku belum yakin maksud kamu 🤔\n"
                            "Kamu bisa coba:\n"
                            "• Cari resep: 'mau masak ayam goreng'\n"
                            "• Filter kondisi: 'resep untuk diabetes'\n"
                )

            # ── 6. Entity extraction ──────────────────────────────────
            entities = self.ner_extractor.extract_all(normalized)

            # ── 7. Slot validation untuk intent pencarian resep ───────
            if intent in self.RECIPE_INTENTS:
                val = self._validate_recipe_slots(intent, entities)
                if val["needs_clarification"]:
                    return self._response(
                        status=self.STATUS_CLARIFICATION,
                        intent=intent,
                        confidence=confidence,
                        entities=entities,
                        action=self.ACTION_ASK_CLARIFICATION,
                        message=val["message"],
                    )

            # ── 8. Validasi keamanan kondisi kesehatan ────────────────
            if entities.get("health_conditions"):
                safety = self._validate_health_safety(entities)
                if not safety["is_safe"]:
                    return self._response(
                        status=self.STATUS_FALLBACK,
                        intent=intent,
                        confidence=confidence,
                        entities=entities,
                        action=self.ACTION_ASK_CLARIFICATION,
                        message=safety["message"],
                    )

            # ── 9. Semua validasi lolos ───────────────────────────────
            action = self.INTENT_ACTION_MAP.get(intent, self.ACTION_MATCH_RECIPE)
            message = self._build_ok_message(intent, entities)

            return self._response(
                status=self.STATUS_OK,
                intent=intent,
                confidence=confidence,
                entities=entities,
                action=action,
                message=message,
            )

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return self._response(
                status=self.STATUS_FALLBACK,
                intent="error",
                confidence=0.0,
                entities={},
                action=self.ACTION_REJECT_INPUT,
                message="Terjadi kesalahan sistem. Silakan coba lagi.",
            )

    # ──────────────────────────────────────────────────────────────────
    # Fast-track untuk input pendek / kata kunci tunggal
    # ──────────────────────────────────────────────────────────────────

    def _fast_track(self, text_lower: str) -> Optional[Dict]:
        words = text_lower.split()

        # Single ingredient → langsung cari_resep
        if len(words) == 1 and words[0] in self.SIMPLE_INGREDIENT_KEYWORDS:
            ingredient = words[0]
            return self._response(
                status=self.STATUS_OK,
                intent="cari_resep",
                confidence=0.75,
                entities={"ingredients": {"main": [ingredient], "avoid": []},
                          "cooking_methods": [], "health_conditions": [],
                          "taste_preferences": [], "time_constraint": None, "region": None},
                action=self.ACTION_MATCH_RECIPE,
                message=f"Mencari resep dengan bahan {ingredient}...",
            )

        # Kata kunci intent tunggal
        for word in words:
            if word in self.SIMPLE_INTENT_KEYWORDS:
                intent = self.SIMPLE_INTENT_KEYWORDS[word]
                # Ekstrak entities kalau ada
                entities = self.ner_extractor.extract_all(text_lower)
                action = self.INTENT_ACTION_MAP.get(intent, self.ACTION_MATCH_RECIPE)
                return self._response(
                    status=self.STATUS_OK,
                    intent=intent,
                    confidence=0.70,
                    entities=entities,
                    action=action,
                    message=self._build_ok_message(intent, entities),
                )

        return None

    # ──────────────────────────────────────────────────────────────────
    # Slot validation
    # ──────────────────────────────────────────────────────────────────

    def _validate_recipe_slots(self, intent: str, entities: Dict) -> Dict:
        """
        Untuk cari_resep / cari_resep_sehat:
        Minimal harus ada salah satu dari: bahan utama, kondisi kesehatan, atau region.
        """
        main_ing = entities.get("ingredients", {}).get("main", [])
        health_cond = entities.get("health_conditions", [])
        region = entities.get("region")
        time_c = entities.get("time_constraint")

        has_enough_info = bool(main_ing or health_cond or region or time_c)

        if not has_enough_info:
            return {
                "needs_clarification": True,
                "message": (
                    "Kamu mau masak apa? Bisa sebutkan:\n"
                    "• Bahan utama (ayam, ikan, tempe, dll.)\n"
                    "• Kondisi kesehatan (diabetes, kolesterol, dll.)\n"
                    "• Asal daerah masakan (Padang, Jawa, Bali, dll.)\n"
                    "• Atau waktu memasak yang diinginkan 😊"
                ),
            }

        return {"needs_clarification": False, "message": ""}

    def _validate_health_safety(self, entities: Dict) -> Dict:
        """
        Periksa konflik antara bahan yang diinginkan dan kondisi kesehatan.
        """
        health_conds = entities.get("health_conditions", [])
        main_ing = entities.get("ingredients", {}).get("main", [])

        conflicts = []
        for cond in health_conds:
            restricted = self.ner_extractor.CONDITION_RESTRICTIONS.get(cond, [])
            for ing in main_ing:
                if ing in restricted:
                    conflicts.append((ing, cond))

        if conflicts:
            msgs = [f"'{ing}' tidak dianjurkan untuk {cond}" for ing, cond in conflicts]
            return {
                "is_safe": False,
                "message": (
                    "⚠️ Perhatian:\n" + "\n".join(f"• {m}" for m in msgs) +
                    "\n\nMaukah kamu mencari alternatif resep yang lebih aman?"
                ),
            }

        return {"is_safe": True, "message": ""}

    # ──────────────────────────────────────────────────────────────────
    # Gibberish detection
    # ──────────────────────────────────────────────────────────────────

    def _is_gibberish(self, text: str) -> bool:
        # Angka saja bukan gibberish (bisa nomor resep)
        if text.strip().isdigit():
            return False

        # Terlalu pendek
        if len(text.strip()) < 2:
            return True

        # Rasio alfabet
        alpha = sum(c.isalpha() for c in text)
        total = len(text.replace(" ", ""))
        if total == 0 or (alpha / total) < 0.25:
            return True

        # Cek kosakata yang dikenal
        words = [w for w in text.split() if len(w) >= 2]
        if not words:
            return True

        recognized = sum(1 for w in words if w in self.KNOWN_VOCAB)
        ratio = recognized / len(words)

        # Kata tunggal yang dikenal → bukan gibberish
        if len(words) == 1 and recognized == 1:
            return False

        return ratio < 0.15

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _build_ok_message(self, intent: str, entities: Dict) -> str:
        main_ing = entities.get("ingredients", {}).get("main", [])
        avoid = entities.get("ingredients", {}).get("avoid", [])
        health = entities.get("health_conditions", [])
        region = entities.get("region")
        time_c = entities.get("time_constraint")

        parts = []
        if main_ing:
            parts.append(f"bahan: {', '.join(main_ing)}")
        if health:
            parts.append(f"kondisi: {', '.join(health)}")
        if region:
            parts.append(f"daerah: {region}")
        if time_c:
            parts.append(f"≤{time_c} menit")
        if avoid:
            parts.append(f"tanpa: {', '.join(avoid)}")

        summary = " | ".join(parts) if parts else "preferensi umum"

        messages = {
            "cari_resep": f"🔍 Mencari resep ({summary})...",
            "cari_resep_sehat": f"🥗 Mencari resep sehat ({summary})...",
            "filter_bahan": f"🔄 Memperbarui filter bahan ({summary})...",
            "filter_waktu": f"⏱️ Mencari resep cepat ({summary})...",
            "filter_region": f"🗺️ Mencari masakan daerah ({summary})...",
            "lihat_detail": "📖 Menampilkan detail resep...",
            "tanya_pantangan": "📋 Menampilkan informasi pantangan makanan...",
            "chitchat": "👋 Hai! Ada yang bisa aku bantu?",
        }
        return messages.get(intent, "Memproses permintaan...")

    def _response(
        self,
        status: str,
        intent: str,
        confidence: float,
        entities: Dict,
        action: str,
        message: str,
    ) -> Dict:
        """Buat response terstruktur sesuai kolom user_queries."""
        return {
            "status": status,           # user_queries.status
            "intent": intent,           # user_queries.intent
            "confidence": round(confidence, 2),  # user_queries.confidence
            "entities": self._clean_entities(entities),  # user_queries.entities
            "action": action,           # untuk layer conversational
            "message": message,         # untuk ditampilkan ke user
        }

    def _clean_entities(self, entities: Dict) -> Dict:
        """
        Bersihkan entities – hanya sertakan field yang tidak kosong.
        Struktur disesuaikan dengan apa yang akan disimpan ke user_queries.entities (JSON).
        """
        cleaned = {}

        # ingredients → ingredients.nama
        ing = entities.get("ingredients", {})
        cleaned["ingredients"] = {
            "main": ing.get("main", []),
            "avoid": ing.get("avoid", []),
        }

        # cooking_methods
        if entities.get("cooking_methods"):
            cleaned["cooking_methods"] = entities["cooking_methods"]

        # health_conditions → health_conditions.nama
        if entities.get("health_conditions"):
            cleaned["health_conditions"] = entities["health_conditions"]

        # taste_preferences
        if entities.get("taste_preferences"):
            cleaned["taste_preferences"] = entities["taste_preferences"]

        # time_constraint → recipes.waktu_masak (integer menit)
        if entities.get("time_constraint") is not None:
            cleaned["time_constraint"] = entities["time_constraint"]

        # region → recipes.region
        if entities.get("region"):
            cleaned["region"] = entities["region"]

        return cleaned


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = EnhancedNLPEngine()

    test_cases = [
        "mau masak ayam goreng",
        "aku diabetes ga boleh gula, carikan resep sehat",
        "resep padang yang tidak terlalu pedas",
        "carikan yang cepat 20 menit",
        "lihat resep nomor 1",
        "berapa kalori rendang",
        "pantangan makanan untuk hipertensi",
        "halo selamat pagi",
        "xzqw abcd efgh",               # gibberish
        "ayam",                           # single keyword
        "mau masak tapi tidak mau santan untuk penderita kolesterol",
    ]

    for q in test_cases:
        print(f"\n{'='*60}")
        print(f"Input    : {q}")
        result = engine.process(q)
        print(f"Status   : {result['status']}")
        print(f"Intent   : {result['intent']} ({result['confidence']})")
        print(f"Action   : {result['action']}")
        print(f"Message  : {result['message']}")
        if result["entities"].get("ingredients", {}).get("main"):
            print(f"Bahan    : {result['entities']['ingredients']['main']}")
        if result["entities"].get("health_conditions"):
            print(f"Kondisi  : {result['entities']['health_conditions']}")
        if result["entities"].get("region"):
            print(f"Region   : {result['entities']['region']}")