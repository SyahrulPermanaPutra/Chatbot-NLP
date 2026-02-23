# src/conversational_ai.py
# Conversational AI – mesin percakapan dua arah dengan slot-filling
#
# Tugas Python:
#   1. Kelola slot-filling multi-turn (kumpulkan bahan, kondisi, region, dll.)
#   2. Tanya pertanyaan yang natural jika info kurang
#   3. Akumulasi context lintas turn
#   4. Hasilkan respons chatbot dalam Bahasa Indonesia
#
# Tugas Laravel:
#   - Query resep ke DB, auth, bisnis logic

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.enhanced_nlp_engine import EnhancedNLPEngine


# ─────────────────────────────────────────────────────────────────────────────
class ConversationContext:
    """State percakapan per sesi (akumulasi lintas turn)."""

    def __init__(self, session_id: str):
        self.session_id   = session_id
        self.history: List[Dict] = []
        self.created_at   = datetime.now()
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
        self.asked_ingredient  = False
        self.asked_health      = False
        self.consecutive_unk   = 0

    def add_turn(self, user_msg: str, bot_msg: str, nlp: Dict):
        self.history.append({
            "ts": datetime.now().isoformat(),
            "user": user_msg,
            "bot": bot_msg,
            "nlp": nlp,
        })
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def update_entities(self, new: Dict):
        ing = new.get("ingredients", {})
        for item in ing.get("main", []):
            if item and item not in self.collected_entities["ingredients"]["main"]:
                self.collected_entities["ingredients"]["main"].append(item)
        for item in ing.get("avoid", []):
            if item and item not in self.collected_entities["ingredients"]["avoid"]:
                self.collected_entities["ingredients"]["avoid"].append(item)

        for key in ("cooking_methods", "health_conditions", "taste_preferences"):
            for val in new.get(key, []):
                if val and val not in self.collected_entities[key]:
                    self.collected_entities[key].append(val)

        if new.get("time_constraint") is not None:
            self.collected_entities["time_constraint"] = new["time_constraint"]
        if new.get("region"):
            self.collected_entities["region"] = new["region"]

    def has_enough_for_search(self) -> bool:
        col = self.collected_entities
        return bool(
            col["ingredients"]["main"]
            or col["health_conditions"]
            or col["region"]
            or col["time_constraint"] is not None
        )

    def clear(self):
        sid = self.session_id
        self.__init__(sid)


# ─────────────────────────────────────────────────────────────────────────────
class ConversationalAI:

    CHITCHAT: Dict[str, str] = {
        "halo":          "Halo! 👋 Aku Kala Rasa, asisten resep masakanmu. Mau masak apa hari ini?",
        "hai":           "Hai! 😊 Mau masak apa hari ini?",
        "selamat pagi":  "Selamat pagi! ☀️ Sarapan apa nih yang mau dimasak?",
        "selamat siang": "Selamat siang! 🌤️ Mau masak makan siang apa?",
        "selamat malam": "Selamat malam! 🌙 Mau masak makan malam apa?",
        "terima kasih":  "Sama-sama! 😊 Semoga masakannya enak ya!",
        "makasih":       "Sama-sama! 🍳 Selamat memasak!",
        "bye":           "Sampai jumpa! 👋 Selamat memasak!",
        "dadah":         "Dadah! 👋 Sampai ketemu lagi!",
        "ok":            "Oke! 😊 Ada yang lain yang bisa aku bantu?",
        "oke":           "Oke! 😊 Ada yang lain yang bisa aku bantu?",
        "tidak ada":     "Oke, tanpa pantangan! Sedang mencari resep terbaik untukmu... 🍳",
        "ga ada":        "Oke, tanpa pantangan! Sedang mencari resep terbaik untukmu... 🍳",
        "ngga ada":      "Oke, tanpa pantangan! Sedang mencari resep terbaik untukmu... 🍳",
    }

    def __init__(self, model_dir: str = "models"):
        self.nlp_engine = EnhancedNLPEngine(model_dir=model_dir)
        self.conversations: Dict[str, ConversationContext] = {}
        print("✓ ConversationalAI ready")

    def _get_ctx(self, session_id: str) -> ConversationContext:
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(session_id)
        return self.conversations[session_id]

    def reset_context(self, session_id: str):
        if session_id in self.conversations:
            self.conversations[session_id].clear()

    # ── Main ──────────────────────────────────────────────────────────

    def process_message(self, session_id: str, message: str) -> Dict:
        ctx = self._get_ctx(session_id)
        nlp = self.nlp_engine.process(message)

        # ── Chitchat langsung ─────────────────────────────────────────
        if self._is_chitchat(message, nlp):
            resp, sugs = self._handle_chitchat(message, ctx)
            ctx.add_turn(message, resp, nlp)
            return self._result(nlp, ctx, resp, sugs)

        # ── Update context ────────────────────────────────────────────
        if nlp.get("entities"):
            ctx.update_entities(nlp["entities"])
            ctx.consecutive_unk = 0

        ctx.current_intent = nlp["intent"]

        # ── Non-recipe intents ────────────────────────────────────────
        NON_RECIPE = {
            "lihat_detail", "tambah_favorit", "hapus_favorit",
            "lihat_favorit", "tanya_pantangan", "tanya_nutrisi",
        }
        if nlp["intent"] in NON_RECIPE:
            resp, sugs = self._handle_non_recipe(nlp, ctx)
            ctx.add_turn(message, resp, nlp)
            return self._result(nlp, ctx, resp, sugs)

        # ── Fallback ──────────────────────────────────────────────────
        if nlp["status"] == "fallback":
            ctx.consecutive_unk += 1
            resp, sugs = self._handle_fallback(nlp, ctx)
            ctx.add_turn(message, resp, nlp)
            return self._result(nlp, ctx, resp, sugs)

        # ── Slot-filling ──────────────────────────────────────────────
        if ctx.has_enough_for_search():
            resp, sugs = self._confirm_search(ctx)
            ctx.add_turn(message, resp, nlp)
            return self._result(nlp, ctx, resp, sugs)

        resp, sugs = self._ask_more(ctx, nlp)
        ctx.add_turn(message, resp, nlp)
        return self._result(nlp, ctx, resp, sugs)

    # ── Handlers ─────────────────────────────────────────────────────

    def _is_chitchat(self, message: str, nlp: Dict) -> bool:
        if nlp["intent"] == "chitchat":
            return True
        msg = message.lower().strip()
        return any(kw in msg for kw in self.CHITCHAT)

    def _handle_chitchat(self, message: str, ctx: ConversationContext) -> Tuple[str, List[str]]:
        msg = message.lower().strip()

        # User menjawab "tidak ada" saat ditanya kondisi kesehatan
        if ctx.pending_question == "health" and any(
            kw in msg for kw in ("tidak ada", "ga ada", "ngga ada", "tidak", "ga", "tidak punya")
        ):
            ctx.pending_question = None
            # Context sudah punya bahan → siap search
            if ctx.has_enough_for_search():
                return self._confirm_search(ctx)
            return (
                "Oke, tanpa pantangan! 😊 Mau menambahkan filter lain?\n"
                "(Contoh: waktu masak, region, preferensi rasa – atau ketik 'cari sekarang')",
                ["Cari sekarang", "Tambah filter waktu", "Tambah region"],
            )

        # Match keyword chitchat
        for kw, reply in self.CHITCHAT.items():
            if kw in msg:
                # Jika sudah punya context, arahkan ke pencarian
                if ctx.has_enough_for_search() and kw in ("ok", "oke"):
                    return self._confirm_search(ctx)
                return reply, self._default_sugs(ctx)

        return (
            "Hai! 😊 Ada yang bisa aku bantu soal resep masakan?",
            self._default_sugs(ctx),
        )

    def _handle_non_recipe(self, nlp: Dict, ctx: ConversationContext) -> Tuple[str, List[str]]:
        messages = {
            "lihat_detail":    ("📖 Memuat detail resep...",                  ["Simpan ke favorit", "Cari resep lain"]),
            "tambah_favorit":  ("❤️ Menyimpan resep ke favoritmu...",         ["Lihat favorit", "Cari resep lain"]),
            "hapus_favorit":   ("🗑️ Menghapus dari favorit...",               ["Lihat favorit", "Cari resep baru"]),
            "lihat_favorit":   ("⭐ Memuat daftar resep favoritmu...",        ["Cari resep baru"]),
            "tanya_nutrisi":   ("🔬 Menampilkan informasi nutrisi...",         ["Kembali"]),
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

        return messages.get(nlp["intent"], (nlp["message"], []))

    def _handle_fallback(self, nlp: Dict, ctx: ConversationContext) -> Tuple[str, List[str]]:
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
            self._default_sugs(ctx),
        )

    def _ask_more(self, ctx: ConversationContext, nlp: Dict) -> Tuple[str, List[str]]:
        """Tanya slot yang belum terisi."""

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

        # Sudah punya bahan & sudah tanya kondisi → seharusnya ready
        return self._confirm_search(ctx)

    def _confirm_search(self, ctx: ConversationContext) -> Tuple[str, List[str]]:
        """Bangun pesan konfirmasi siap search."""
        col   = ctx.collected_entities
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

    def _default_sugs(self, ctx: ConversationContext) -> List[str]:
        if not ctx.collected_entities["ingredients"]["main"]:
            return ["Mau masak ayam", "Resep ikan", "Masakan sehat", "Lihat favorit"]
        return ["Cari resep lain", "Filter waktu", "Simpan favorit", "Lihat favorit"]

    def _result(self, nlp: Dict, ctx: ConversationContext, response: str, suggestions: List[str]) -> Dict:
        return {
            "nlp_result": {
                "status":     nlp["status"],
                "intent":     nlp["intent"],
                "confidence": nlp["confidence"],
                "entities":   nlp["entities"],
                "action":     nlp["action"],
                "message":    nlp["message"],
            },
            "context": {
                "collected_entities":    ctx.collected_entities,
                "conversation_turns":    len(ctx.history),
                "current_intent":        ctx.current_intent,
                "has_enough_for_search": ctx.has_enough_for_search(),
            },
            "response":    response,
            "suggestions": suggestions,
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Multi-turn Chatbot Test ===\n")
    ai  = ConversationalAI()
    sid = "demo_session"

    turns = [
        "halo",
        "mau masak",
        "ayam",
        "tidak ada",
        "yang tidak pedas dan cepat 30 menit",
        "lihat resep 1",
        "simpan ke favorit",
        "terima kasih",
    ]

    for msg in turns:
        print(f"\nUser : {msg}")
        r = ai.process_message(sid, msg)
        print(f"Bot  : {r['response']}")
        if r["suggestions"]:
            print(f"Saran: {r['suggestions']}")
        nlp = r["nlp_result"]
        ctx = r["context"]
        print(f"[intent={nlp['intent']} | status={nlp['status']}]")
        if ctx["collected_entities"]["ingredients"]["main"]:
            print(f"[ctx.bahan={ctx['collected_entities']['ingredients']['main']}]")
        print(f"[ready={ctx['has_enough_for_search']}]")