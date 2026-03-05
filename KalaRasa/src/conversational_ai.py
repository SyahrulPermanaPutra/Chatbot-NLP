# src/conversational_ai.py
# ─────────────────────────────────────────────────────────────────────────────
# Conversational AI — mesin percakapan multi-turn untuk Kala Rasa Chatbot
#
# Tanggung jawab modul ini (Python layer):
#   • Kelola slot-filling lintas turn (bahan, kondisi, region, dll.)
#   • Topic-switch detection agar entity baru me-replace entity lama
#   • Hasilkan bot response + suggestion dalam Bahasa Indonesia
#
# Tanggung jawab Laravel layer:
#   • Auth, DB query, CBR matching, simpan log ke DB
#
# Pipeline per pesan:
#   User Message
#     → [1] NLP Engine (intent + entity extraction)
#     → [2] Chitchat Guard          ← cek dulu sebelum apapun
#     → [3] Context Update          ← topic-switch aware REPLACE/APPEND
#     → [4] Intent Router           ← non-recipe / fallback / slot-filling
#     → [5] Response Builder        ← _result() dengan get_search_entities()
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.enhanced_nlp_engine import EnhancedNLPEngine

logger = logging.getLogger(__name__)

# =============================================================================
# KEBIJAKAN ENTITY UPDATE
# =============================================================================
# ┌────────────────────┬──────────────────────────────────────────────────────┐
# │ Entity             │ Strategi Update                                      │
# ├────────────────────┼──────────────────────────────────────────────────────┤
# │ ingredients.main   │ REPLACE jika user sebut bahan baru (topic switch)    │
# │ ingredients.avoid  │ APPEND selalu (pantangan bersifat akumulatif)        │
# │ health_conditions  │ REPLACE jika ada kondisi baru yang eksplisit         │
# │ taste_preferences  │ REPLACE jika disebutkan baru                         │
# │ cooking_methods    │ APPEND (bisa kombinasi)                              │
# │ time_constraint    │ REPLACE (override)                                   │
# │ region             │ REPLACE (override)                                   │
# └────────────────────┴──────────────────────────────────────────────────────┘

# =============================================================================
# PANDUAN CHIT-CHAT (baca sebelum mengubah logic ini)
# =============================================================================
# Chit-chat dideteksi lewat DUA layer secara berurutan:
#
#   Layer 1 — NLP Engine (intent_classifier.py):
#     Jika teks pendek & tidak mengandung kata kunci resep,
#     classifier mungkin mengembalikan intent="chitchat".
#     Ini adalah jalur IDEAL karena akurat berbasis ML.
#
#   Layer 2 — Keyword Guard (_is_chitchat / CHITCHAT_KEYWORDS):
#     Fallback rule-based untuk kata-kata yang PASTI chit-chat
#     terlepas dari hasil classifier (mis. "halo", "bye").
#     PENTING: keyword harus SANGAT spesifik agar tidak false-positive
#     pada pesan resep (misal "tidak ada" bisa berarti pantangan).
#
# BUG SUMBER CHIT-CHAT YANG SALAH:
#   1. Keyword terlalu luas → "tidak" menangkap "tidak pedas"
#   2. Substring match pada kalimat panjang → "ok" menangkap "okok tolong"
#   3. pending_question tidak di-reset → jawaban pantangan dikira chit-chat
#   4. CHITCHAT dict mengandung frase yang juga muncul di konteks resep
#
# ATURAN PENAMBAHAN CHITCHAT_KEYWORDS:
#   ✅ Boleh: kata/frase yang TIDAK MUNGKIN ada dalam konteks resep
#   ❌ Jangan: kata ambigu seperti "tidak", "ga", "ya", "oke" tanpa konteks
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
class ConversationContext:
    """State percakapan per sesi — topic-switch aware."""

    def __init__(self, session_id: str):
        self.session_id  = session_id
        self.history: List[Dict] = []
        self.created_at  = datetime.now()
        self.current_intent: Optional[str] = None

        self.collected_entities: Dict = {
            "ingredients":       {"main": [], "avoid": []},
            "cooking_methods":   [],
            "health_conditions": [],
            "taste_preferences": [],
            "time_constraint":   None,
            "region":            None,
        }

        # Slot-filling state
        self.pending_question: Optional[str] = None
        self.asked_ingredient = False
        self.asked_health     = False
        self.consecutive_unk  = 0

        # Snapshot entity per-turn (untuk logging & debugging leakage)
        self.last_turn_entities: Dict = {}

    # ── History ──────────────────────────────────────────────────────────────

    def add_turn(self, user_msg: str, bot_msg: str, nlp: Dict) -> None:
        self.history.append({
            "ts":   datetime.now().isoformat(),
            "user": user_msg,
            "bot":  bot_msg,
            "nlp":  nlp,
        })
        if len(self.history) > 50:
            self.history = self.history[-50:]

    # ── Entity management — topic-switch aware ───────────────────────────────

    def update_entities(self, new: Dict) -> Dict:
        """
        Update collected_entities dari entity pesan terbaru.

        Menerapkan kebijakan REPLACE vs APPEND per entity type.
        Mendeteksi topic switch pada ingredients.main.

        Returns:
            dict berisi entity yang di-replace untuk logging.
        """
        replaced: Dict = {}
        ing      = new.get("ingredients", {})
        new_main = [i for i in ing.get("main", []) if i]

        # ingredients.main — REPLACE jika bahan berubah (topic switch)
        if new_main:
            old_main = self.collected_entities["ingredients"]["main"][:]
            if old_main and set(old_main) != set(new_main):
                replaced["ingredients_main"] = {"old": old_main, "new": new_main}
                logger.info(
                    "[CTX][%s] topic-switch: %s → %s",
                    self.session_id, old_main, new_main,
                )
                # Reset slot-filling state terkait ingredient lama
                self.asked_health     = False
                self.pending_question = None
            self.collected_entities["ingredients"]["main"] = new_main

        # ingredients.avoid — APPEND (akumulatif)
        for item in [i for i in ing.get("avoid", []) if i]:
            if item not in self.collected_entities["ingredients"]["avoid"]:
                self.collected_entities["ingredients"]["avoid"].append(item)

        # cooking_methods — APPEND
        for val in new.get("cooking_methods", []):
            if val and val not in self.collected_entities["cooking_methods"]:
                self.collected_entities["cooking_methods"].append(val)

        # health_conditions — REPLACE jika ada kondisi baru
        new_conds = [c for c in new.get("health_conditions", []) if c]
        if new_conds:
            old = self.collected_entities["health_conditions"][:]
            if old != new_conds:
                replaced["health_conditions"] = {"old": old, "new": new_conds}
            self.collected_entities["health_conditions"] = new_conds

        # taste_preferences — REPLACE
        new_tastes = [t for t in new.get("taste_preferences", []) if t]
        if new_tastes:
            self.collected_entities["taste_preferences"] = new_tastes

        # time_constraint — REPLACE
        if new.get("time_constraint") is not None:
            self.collected_entities["time_constraint"] = new["time_constraint"]

        # region — REPLACE
        if new.get("region"):
            self.collected_entities["region"] = new["region"]

        self.last_turn_entities = new
        return replaced

    def get_search_entities(self) -> Dict:
        """
        Kembalikan snapshot bersih entities untuk CBR matching.

        Ini adalah satu-satunya sumber kebenaran yang dikirim ke Laravel
        sebagai 'context_entities'. Mengembalikan deep copy sehingga aman
        dari mutation tidak sengaja di layer atas.

        PENTING: Laravel wajib menggunakan nilai ini — jangan merge mandiri
        antara nlp['entities'] dan context lama.
        """
        return {
            "ingredients": {
                "main":  self.collected_entities["ingredients"]["main"][:],
                "avoid": self.collected_entities["ingredients"]["avoid"][:],
            },
            "cooking_methods":   self.collected_entities["cooking_methods"][:],
            "health_conditions": self.collected_entities["health_conditions"][:],
            "taste_preferences": self.collected_entities["taste_preferences"][:],
            "time_constraint":   self.collected_entities["time_constraint"],
            "region":            self.collected_entities["region"],
        }

    def has_enough_for_search(self) -> bool:
        """Cek apakah context sudah cukup untuk melakukan CBR search."""
        col = self.collected_entities
        return bool(
            col["ingredients"]["main"]
            or col["health_conditions"]
            or col["region"]
            or col["time_constraint"] is not None
        )

    def clear(self) -> None:
        """Reset semua state percakapan ke kondisi awal."""
        sid = self.session_id
        self.__init__(sid)
        self.last_turn_entities = {}


# ─────────────────────────────────────────────────────────────────────────────
class ConversationalAI:
    """
    Mesin percakapan utama — mengelola session, routing intent, dan slot-filling.

    Cara pakai:
        ai = ConversationalAI()
        result = ai.process_message(session_id, message, reset=False)
        # result["context"]["collected_entities"] → kirim ke Laravel untuk CBR
    """

    # ── Chit-chat responses ───────────────────────────────────────────────────
    # PENTING: keyword di sini adalah EXACT substring match pada lowercased msg.
    # Jangan tambahkan kata ambigu. Gunakan frase yang cukup spesifik.
    # Untuk kata yang ambigu (misal "tidak", "ga"), tangani via pending_question
    # di _handle_pending_health_answer(), bukan di sini.
    CHITCHAT_RESPONSES: Dict[str, str] = {
        # Salam
        "halo":          "Halo! 👋 Aku Kala Rasa, asisten resep masakanmu. Mau masak apa hari ini?",
        "hai":           "Hai! 😊 Mau masak apa hari ini?",
        "selamat pagi":  "Selamat pagi! ☀️ Sarapan apa nih yang mau dimasak?",
        "selamat siang": "Selamat siang! 🌤️ Mau masak makan siang apa?",
        "selamat malam": "Selamat malam! 🌙 Mau masak makan malam apa?",
        # Terima kasih / penutup
        "terima kasih":  "Sama-sama! 😊 Semoga masakannya enak ya!",
        "makasih":       "Sama-sama! 🍳 Selamat memasak!",
        "bye":           "Sampai jumpa! 👋 Selamat memasak!",
        "dadah":         "Dadah! 👋 Sampai ketemu lagi!",
        # Konfirmasi pendek — hanya aktif setelah context penuh (lihat _handle_short_confirm)
        "ok":            "__CONFIRM__",
        "oke":           "__CONFIRM__",
        "siap":          "__CONFIRM__",
        "lanjut":        "__CONFIRM__",
        # Jawaban tidak ada pantangan — ditangani via _handle_pending_health_answer
        "tidak ada":     "__NO_HEALTH__",
        "ga ada":        "__NO_HEALTH__",
        "ngga ada":      "__NO_HEALTH__",
        "tidak punya":   "__NO_HEALTH__",
        "gak ada":       "__NO_HEALTH__",
    }

    # Intent yang tidak perlu slot-filling resep
    NON_RECIPE_INTENTS = frozenset({
        "lihat_detail", "tambah_favorit", "hapus_favorit",
        "lihat_favorit", "tanya_pantangan", "tanya_nutrisi",
    })

    def __init__(self, model_dir: str = "models"):
        self.nlp_engine    = EnhancedNLPEngine(model_dir=model_dir)
        self.conversations: Dict[str, ConversationContext] = {}
        logger.info("ConversationalAI ready (model_dir=%s)", model_dir)
        print("✓ ConversationalAI ready")

    # ── Session management ────────────────────────────────────────────────────

    def _get_ctx(self, session_id: str) -> ConversationContext:
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(session_id)
        return self.conversations[session_id]

    def reset_context(self, session_id: str) -> None:
        """Reset state percakapan untuk session tertentu."""
        if session_id in self.conversations:
            self.conversations[session_id].clear()
            logger.info("[SESSION][%s] context reset", session_id)

    # ── Main entry point ──────────────────────────────────────────────────────

    def process_message(self, session_id: str, message: str, reset: bool = False) -> Dict:
        """
        Proses satu pesan user dan kembalikan response lengkap.

        Pipeline:
          [1] Reset atomik jika diminta
          [2] NLP Engine — intent + entity dari pesan ini
          [3] Chitchat Guard — tangani chit-chat SEBELUM update context
          [4] Context Update — REPLACE/APPEND entity berdasarkan kebijakan
          [5] Intent Router — non-recipe | fallback | slot-filling
          [6] Response Builder — _result() dengan get_search_entities()

        Args:
            session_id: ID unik sesi percakapan
            message:    Pesan dari user
            reset:      True untuk mulai sesi baru (hapus context lama)

        Returns:
            Dict dengan keys: nlp_result, context, response, suggestions
        """
        # [1] Reset atomik — harus sebelum _get_ctx agar context baru
        if reset:
            self.reset_context(session_id)

        ctx = self._get_ctx(session_id)
        nlp = self.nlp_engine.process(message)

        logger.debug(
            "[NLP][%s] intent=%s conf=%.2f entities_main=%s",
            session_id,
            nlp.get("intent"),
            nlp.get("confidence", 0),
            nlp.get("entities", {}).get("ingredients", {}).get("main", []),
        )

        # [3] Chitchat Guard — cek SEBELUM update context
        # Mengapa di sini: chit-chat tidak boleh mengubah entity context
        chitchat_result = self._try_handle_chitchat(message, nlp, ctx)
        if chitchat_result is not None:
            resp, sugs = chitchat_result
            ctx.add_turn(message, resp, nlp)
            return self._result(nlp, ctx, resp, sugs)

        # [4] Context Update — setelah guard chit-chat lolos
        replaced = {}
        if nlp.get("entities"):
            replaced = ctx.update_entities(nlp["entities"])
            ctx.consecutive_unk = 0

        if replaced:
            logger.info("[CTX][%s] entities replaced: %s", session_id, replaced)

        logger.debug(
            "[CTX][%s] after update main=%s",
            session_id,
            ctx.collected_entities["ingredients"]["main"],
        )

        ctx.current_intent = nlp["intent"]

        # [5] Intent Router
        if nlp["intent"] in self.NON_RECIPE_INTENTS:
            resp, sugs = self._handle_non_recipe(nlp, ctx)
        elif nlp["status"] == "fallback":
            ctx.consecutive_unk += 1
            resp, sugs = self._handle_fallback(nlp, ctx)
        elif ctx.has_enough_for_search():
            resp, sugs = self._confirm_search(ctx)
        else:
            resp, sugs = self._ask_more(ctx, nlp)

        ctx.add_turn(message, resp, nlp)
        return self._result(nlp, ctx, resp, sugs)

    # ── Chitchat Guard — dipisah dari handler ────────────────────────────────

    def _try_handle_chitchat(
        self, message: str, nlp: Dict, ctx: ConversationContext
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Cek apakah pesan ini adalah chit-chat. Jika ya, kembalikan (resp, sugs).
        Jika bukan, kembalikan None.

        URUTAN PENGECEKAN (penting — jangan ubah urutan):
          1. pending_question handler — jawaban user atas pertanyaan bot
          2. NLP intent == 'chitchat' — dari ML classifier (akurat)
          3. CHITCHAT_RESPONSES keyword — rule-based fallback (ketat)

        Kenapa pending_question dicek pertama:
          Saat bot bertanya "ada pantangan?" dan user jawab "tidak ada",
          kata "tidak ada" ada di CHITCHAT_RESPONSES. Tanpa pengecekan
          pending_question lebih dulu, jawaban ini akan dianggap chit-chat
          biasa, bukan jawaban slot-filling. Ini adalah bug yang paling
          sering terjadi pada chatbot hybrid.
        """
        msg_lower = message.lower().strip()

        # 1. Pending question handler (slot-filling answer)
        if ctx.pending_question:
            handled = self._handle_pending_answer(msg_lower, ctx)
            if handled is not None:
                return handled
            # Jika pending_question ada tapi jawaban tidak cocok,
            # teruskan ke pengecekan normal (user mungkin menjawab hal lain)

        # 2. NLP ML classifier mendeteksi chitchat
        if nlp.get("intent") == "chitchat":
            return self._dispatch_chitchat_keyword(msg_lower, ctx)

        # 3. Rule-based keyword guard — HANYA untuk kata yang SANGAT spesifik
        # Cek full-match pada keyword (bukan substring dari kalimat panjang)
        # untuk menghindari false-positive pada pesan resep.
        for kw in self.CHITCHAT_RESPONSES:
            # Gunakan full-match atau word-boundary agar "ok" tidak
            # menangkap "okeh lanjut masak ayam goreng" yang punya entity
            if self._is_chitchat_keyword_match(msg_lower, kw):
                # Exception: jika pesan juga mengandung entity resep yang
                # signifikan, biarkan NLP pipeline menanganinya
                has_recipe_entity = bool(
                    nlp.get("entities", {}).get("ingredients", {}).get("main")
                    or nlp.get("entities", {}).get("health_conditions")
                )
                if has_recipe_entity:
                    return None  # biarkan recipe pipeline yang handle
                return self._dispatch_chitchat_keyword(msg_lower, ctx)

        return None

    def _is_chitchat_keyword_match(self, msg_lower: str, keyword: str) -> bool:
        """
        Match keyword chit-chat dengan aturan yang lebih ketat dari substring biasa.

        Untuk single-word keyword (mis. "halo", "bye"):
          - Cocok jika ada sebagai kata utuh (word boundary)
        Untuk multi-word keyword (mis. "terima kasih", "tidak ada"):
          - Cocok jika msg_lower mengandung exact frase itu
          - Tapi pesan tidak boleh terlalu panjang (> 5 kata berarti ada konteks lain)
        """
        words = msg_lower.split()

        if " " in keyword:
            # Multi-word: exact substring + panjang pesan dibatasi
            return keyword in msg_lower and len(words) <= 6
        else:
            # Single-word: harus cocok sebagai kata (bukan bagian dari kata lain)
            # Contoh: "ok" tidak menangkap "tokok" atau "oke banget mau masak ayam"
            return keyword in words and len(words) <= 3

    def _handle_pending_answer(
        self, msg_lower: str, ctx: ConversationContext
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Tangani jawaban user atas pertanyaan bot yang sedang pending.
        Kembalikan (resp, sugs) jika jawaban cocok, None jika tidak.
        """
        if ctx.pending_question == "health":
            # Deteksi jawaban "tidak ada pantangan"
            no_health_signals = {
                "tidak ada", "ga ada", "ngga ada", "gak ada",
                "tidak punya", "tidak ada pantangan", "ga punya",
                "sehat", "normal", "biasa saja", "biasa aja",
            }
            if any(sig in msg_lower for sig in no_health_signals):
                ctx.pending_question = None
                if ctx.has_enough_for_search():
                    return self._confirm_search(ctx)
                return (
                    "Oke, tanpa pantangan! 😊 Mau menambahkan filter lain?\n"
                    "(Contoh: waktu masak, region – atau ketik 'cari sekarang')",
                    ["Cari sekarang", "Tambah filter waktu", "Tambah region"],
                )
        return None

    def _dispatch_chitchat_keyword(
        self, msg_lower: str, ctx: ConversationContext
    ) -> Tuple[str, List[str]]:
        """
        Temukan response yang tepat untuk keyword chit-chat.
        Menangani sentinel value __CONFIRM__ dan __NO_HEALTH__ secara terpisah.
        """
        # Cari keyword yang cocok
        matched_response = None
        for kw, resp_template in self.CHITCHAT_RESPONSES.items():
            if self._is_chitchat_keyword_match(msg_lower, kw):
                matched_response = resp_template
                break

        if matched_response == "__CONFIRM__":
            # "ok", "oke", "siap", "lanjut" — hanya trigger search jika sudah punya context
            if ctx.has_enough_for_search() and len(ctx.history) > 0:
                return self._confirm_search(ctx)
            return (
                "Oke! 😊 Mau masak apa hari ini?",
                self._default_suggestions(ctx),
            )

        if matched_response == "__NO_HEALTH__":
            # Fallthrough ke _handle_pending_answer sudah ditangani di _try_handle_chitchat
            # Jika sampai sini berarti tidak ada pending_question — tangani sebagai generic
            ctx.pending_question = None
            if ctx.has_enough_for_search():
                return self._confirm_search(ctx)
            return (
                "Oke! Mau masak apa? 😊",
                self._default_suggestions(ctx),
            )

        if matched_response:
            return matched_response, self._default_suggestions(ctx)

        # Default chitchat response
        return (
            "Hai! 😊 Ada yang bisa aku bantu soal resep masakan?",
            self._default_suggestions(ctx),
        )

    # ── Intent handlers ───────────────────────────────────────────────────────

    def _handle_non_recipe(
        self, nlp: Dict, ctx: ConversationContext
    ) -> Tuple[str, List[str]]:
        """Tangani intent yang tidak membutuhkan recipe search."""
        RESPONSE_MAP = {
            "lihat_detail":   ("📖 Memuat detail resep...",             ["Simpan ke favorit", "Cari resep lain"]),
            "tambah_favorit": ("❤️ Menyimpan resep ke favoritmu...",    ["Lihat favorit", "Cari resep lain"]),
            "hapus_favorit":  ("🗑️ Menghapus dari favorit...",          ["Lihat favorit", "Cari resep baru"]),
            "lihat_favorit":  ("⭐ Memuat daftar resep favoritmu...",   ["Cari resep baru"]),
            "tanya_nutrisi":  ("🔬 Menampilkan informasi nutrisi...",    ["Kembali"]),
        }

        if nlp["intent"] == "tanya_pantangan":
            conds = ctx.collected_entities["health_conditions"]
            if conds:
                return (
                    f"📋 Menampilkan pantangan untuk {', '.join(conds)}...",
                    ["Carikan resep yang cocok"],
                )
            return (
                "Kondisi kesehatan apa yang ingin kamu ketahui? 💊",
                ["Diabetes", "Kolesterol", "Hipertensi", "Asam Urat"],
            )

        return RESPONSE_MAP.get(nlp["intent"], (nlp.get("message", "Memproses..."), []))

    def _handle_fallback(
        self, nlp: Dict, ctx: ConversationContext
    ) -> Tuple[str, List[str]]:
        """Tangani pesan yang tidak dipahami sistem."""
        if ctx.consecutive_unk >= 3:
            return (
                "Sepertinya aku kesulitan memahami 😅. Coba salah satu ini:",
                ["Mau masak ayam", "Resep untuk diabetes", "Masakan Padang", "Lihat favorit saya"],
            )
        if not ctx.collected_entities["ingredients"]["main"]:
            return (
                "Mau masak bahan apa hari ini? 🥘",
                ["Ayam", "Ikan", "Tempe/Tahu", "Sayuran", "Daging Sapi"],
            )
        return (
            nlp.get("message", "Maaf, aku kurang paham. Coba ceritakan mau masak apa? 😊"),
            self._default_suggestions(ctx),
        )

    def _ask_more(
        self, ctx: ConversationContext, nlp: Dict
    ) -> Tuple[str, List[str]]:
        """Tanya slot yang belum terisi — slot-filling logic."""
        if not ctx.collected_entities["ingredients"]["main"]:
            ctx.asked_ingredient = True
            return (
                "Mau masak bahan apa hari ini? 🥘\n"
                "Contoh: ayam, ikan, tempe, tahu, atau bahan lainnya?",
                ["Ayam", "Ikan", "Tempe", "Tahu", "Daging Sapi"],
            )

        if not ctx.asked_health and not ctx.collected_entities["health_conditions"]:
            ctx.asked_health     = True
            ctx.pending_question = "health"
            ings = ", ".join(ctx.collected_entities["ingredients"]["main"])
            return (
                f"Oke, punya **{ings}**! 😊\n"
                "Ada kondisi kesehatan atau pantangan tertentu? 💊\n"
                "(Contoh: diabetes, kolesterol – atau ketik 'tidak ada')",
                ["Diabetes", "Kolesterol", "Hipertensi", "Asam Urat", "Tidak ada"],
            )

        # Semua slot terpenuhi — siap search
        return self._confirm_search(ctx)

    def _confirm_search(self, ctx: ConversationContext) -> Tuple[str, List[str]]:
        """Bangun pesan konfirmasi bahwa sistem siap melakukan recipe search."""
        # Gunakan get_search_entities() untuk konsistensi dengan apa yang
        # dikirim ke Laravel — bukan raw collected_entities
        col   = ctx.get_search_entities()
        parts = []

        if col["ingredients"]["main"]:
            parts.append("bahan **" + ", ".join(col["ingredients"]["main"]) + "**")
        if col["health_conditions"]:
            parts.append("cocok untuk **" + ", ".join(col["health_conditions"]) + "**")
        if col["region"]:
            parts.append(f"masakan **{col['region']}**")
        if col["time_constraint"]:
            parts.append(f"≤ **{col['time_constraint']} menit**")
        if col["ingredients"]["avoid"]:
            parts.append("tanpa **" + ", ".join(col["ingredients"]["avoid"]) + "**")
        if col["taste_preferences"]:
            parts.append("rasa **" + ", ".join(col["taste_preferences"]) + "**")

        criteria = " | ".join(parts) if parts else "preferensi umum"
        return (
            f"🔍 Mencari resep dengan: {criteria}...",
            ["Filter waktu ≤30 menit", "Ganti bahan", "Tambah kondisi kesehatan"],
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _default_suggestions(self, ctx: ConversationContext) -> List[str]:
        if not ctx.collected_entities["ingredients"]["main"]:
            return ["Mau masak ayam", "Resep ikan", "Masakan sehat", "Lihat favorit"]
        return ["Cari resep lain", "Filter waktu", "Simpan favorit", "Lihat favorit"]

    def _result(
        self,
        nlp: Dict,
        ctx: ConversationContext,
        response: str,
        suggestions: List[str],
    ) -> Dict:
        """
        Bangun dict hasil yang dikirim kembali ke Flask → Laravel.

        PENTING:
          - 'entities'         = output NLP dari pesan INI saja (untuk logging/debug)
          - 'collected_entities' = get_search_entities() setelah topic-switch detection
                                   INI yang dipakai Laravel untuk CBR matching
        """
        return {
            "nlp_result": {
                "status":     nlp["status"],
                "intent":     nlp["intent"],
                "confidence": nlp["confidence"],
                "entities":   nlp["entities"],   # dari pesan ini saja
                "action":     nlp["action"],
                "message":    nlp["message"],
            },
            "context": {
                # Snapshot bersih setelah topic-switch detection
                # Laravel WAJIB gunakan ini untuk CBR, bukan merge mandiri
                "collected_entities":    ctx.get_search_entities(),
                "conversation_turns":    len(ctx.history),
                "current_intent":        ctx.current_intent,
                "has_enough_for_search": ctx.has_enough_for_search(),
                # Debug only — entitas mentah dari turn ini
                "last_turn_entities":    ctx.last_turn_entities,
            },
            "response":    response,
            "suggestions": suggestions,
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    print("=" * 65)
    print("CHITCHAT & TOPIC SWITCH TEST — Kala Rasa Chatbot")
    print("=" * 65)

    ai  = ConversationalAI()
    sid = "test_session"

    scenarios = [
        # (pesan, label untuk output)
        ("Halo",                                    "Salam pembuka"),
        ("Saya ingin makan ayam",                   "Turn 1: tanya resep ayam"),
        ("tidak ada",                               "Turn 2: jawab tidak ada pantangan"),
        ("Kalau resep tentang kambing apakah ada?", "Turn 3: topic switch ke kambing"),
        ("ok",                                      "Turn 4: konfirmasi (harusnya search kambing)"),
        ("terima kasih",                            "Turn 5: penutup chit-chat"),
        ("mau masak untuk diabetes",                "Turn 6: kembali ke resep + kondisi"),
    ]

    for msg, label in scenarios:
        print(f"\n{'─' * 65}")
        print(f"[{label}]")
        print(f"User    : {msg}")
        r   = ai.process_message(sid, msg)
        ctx = r["context"]
        nlp = r["nlp_result"]
        col = ctx["collected_entities"]
        print(f"Bot     : {r['response']}")
        print(f"Intent  : {nlp['intent']} ({nlp['confidence']:.2f})")
        print(f"CTX main: {col['ingredients']['main']}")
        print(f"CTX health: {col['health_conditions']}")
        print(f"Ready   : {ctx['has_enough_for_search']}")