# src/ner_extractor.py
# Named Entity Recognition – disesuaikan dengan skema database kala_rasa_jtv
#
# Tabel relevan:
#   ingredients          : id, nama, kategori (protein|sayuran|karbohidrat|bumbu|lemak|penyedap), sub_kategori
#   health_conditions    : id, nama
#   health_condition_restrictions: health_condition_id, ingredient_id, severity (hindari|batasi|anjuran)
#   recipes              : id, nama, waktu_masak, region, kategori
#
# Entity types yang diekstrak:
#   ingredients.main     -> bahan utama (cocok dengan ingredients.nama)
#   ingredients.avoid    -> bahan yang dihindari
#   cooking_methods      -> teknik memasak (untuk filter recipes.kategori / langkah_langkah)
#   health_conditions    -> nama kondisi (cocok dengan health_conditions.nama)
#   taste_preferences    -> preferensi rasa
#   time_constraint      -> waktu masak dalam menit (cocok dengan recipes.waktu_masak)
#   region               -> daerah asal masakan (cocok dengan recipes.region)

import re
from typing import Dict, List, Optional


class NERExtractor:
    """
    Rule-based NER extractor yang entitiesnya selaras dengan database kala_rasa_jtv.
    Tidak membutuhkan file JSON eksternal – semua kamus didefinisikan langsung di kelas.
    """

    # ------------------------------------------------------------------
    # KAMUS BAHAN MAKANAN  (sesuai ingredients.nama & ingredients.kategori)
    # ------------------------------------------------------------------
    INGREDIENTS: Dict[str, str] = {
        # ── protein ──────────────────────────────────────────────────
        "ayam": "protein", "daging ayam": "protein",
        "sapi": "protein", "daging sapi": "protein",
        "kambing": "protein", "daging kambing": "protein",
        "babi": "protein",
        "ikan": "protein", "ikan lele": "protein", "ikan nila": "protein",
        "ikan mas": "protein", "ikan tongkol": "protein", "ikan tuna": "protein",
        "ikan salmon": "protein", "ikan kakap": "protein", "ikan mujair": "protein",
        "ikan patin": "protein", "ikan bandeng": "protein",
        "udang": "protein",
        "cumi": "protein", "cumi-cumi": "protein",
        "kepiting": "protein",
        "kerang": "protein",
        "telur": "protein", "telur ayam": "protein", "telur bebek": "protein",
        "tempe": "protein",
        "tahu": "protein",
        "susu": "protein",
        "keju": "protein",
        "yogurt": "protein",

        # ── sayuran ──────────────────────────────────────────────────
        "bayam": "sayuran", "kangkung": "sayuran", "sawi": "sayuran",
        "brokoli": "sayuran", "kol": "sayuran", "kubis": "sayuran",
        "wortel": "sayuran", "buncis": "sayuran", "kacang panjang": "sayuran",
        "tauge": "sayuran", "taog": "sayuran",
        "timun": "sayuran", "ketimun": "sayuran",
        "terong": "sayuran",
        "tomat": "sayuran",
        "labu": "sayuran", "labu siam": "sayuran", "labu kuning": "sayuran",
        "pepaya muda": "sayuran",
        "nangka muda": "sayuran",
        "jagung": "sayuran",
        "pare": "sayuran",
        "daun singkong": "sayuran", "daun pepaya": "sayuran",
        "jamur": "sayuran", "jamur tiram": "sayuran", "jamur kuping": "sayuran",
        "asparagus": "sayuran",
        "paprika": "sayuran",
        "seledri": "sayuran",
        "daun bawang": "sayuran",

        # ── karbohidrat ──────────────────────────────────────────────
        "nasi": "karbohidrat", "beras": "karbohidrat", "beras merah": "karbohidrat",
        "beras putih": "karbohidrat",
        "mie": "karbohidrat", "mi": "karbohidrat", "mie telur": "karbohidrat",
        "mie kuning": "karbohidrat", "mie bihun": "karbohidrat", "bihun": "karbohidrat",
        "kwetiau": "karbohidrat",
        "pasta": "karbohidrat", "spaghetti": "karbohidrat", "makaroni": "karbohidrat",
        "fettuccine": "karbohidrat",
        "roti": "karbohidrat", "roti tawar": "karbohidrat",
        "kentang": "karbohidrat",
        "ubi": "karbohidrat", "ubi jalar": "karbohidrat",
        "singkong": "karbohidrat",
        "tepung": "karbohidrat", "tepung terigu": "karbohidrat",
        "tepung beras": "karbohidrat", "tepung tapioka": "karbohidrat",
        "oat": "karbohidrat",
        "jagung manis": "karbohidrat",

        # ── bumbu ─────────────────────────────────────────────────────
        "bawang merah": "bumbu", "bawang putih": "bumbu", "bawang bombai": "bumbu",
        "cabai": "bumbu", "cabe": "bumbu", "cabai merah": "bumbu",
        "cabai rawit": "bumbu", "cabe rawit": "bumbu",
        "jahe": "bumbu", "kunyit": "bumbu", "lengkuas": "bumbu",
        "serai": "bumbu", "daun salam": "bumbu", "daun jeruk": "bumbu",
        "kemiri": "bumbu", "ketumbar": "bumbu", "jintan": "bumbu",
        "merica": "bumbu", "lada": "bumbu",
        "terasi": "bumbu", "belacan": "bumbu",
        "kencur": "bumbu", "galangal": "bumbu",

        # ── lemak ─────────────────────────────────────────────────────
        "minyak goreng": "lemak", "minyak kelapa": "lemak", "minyak zaitun": "lemak",
        "santan": "lemak", "santan kental": "lemak",
        "mentega": "lemak", "butter": "lemak",
        "margarin": "lemak",
        "kelapa": "lemak", "kelapa parut": "lemak",
        "alpukat": "lemak",

        # ── penyedap ──────────────────────────────────────────────────
        "kecap manis": "penyedap", "kecap asin": "penyedap", "kecap": "penyedap",
        "saus tiram": "penyedap", "saus tomat": "penyedap",
        "garam": "penyedap", "gula": "penyedap", "gula merah": "penyedap",
        "gula pasir": "penyedap",
        "penyedap rasa": "penyedap", "kaldu": "penyedap",
        "cuka": "penyedap", "asam jawa": "penyedap",
        "air jeruk": "penyedap",
    }

    # Sinonim bahan → nama standar database
    INGREDIENT_SYNONYMS: Dict[str, str] = {
        "cabe": "cabai", "cabe rawit": "cabai rawit", "cabe merah": "cabai merah",
        "ayem": "ayam", "aym": "ayam",
        "mie": "mi", "mi instan": "mi",
        "lada": "merica",
        "daun jeruk purut": "daun jeruk",
        "lengkuah": "lengkuas",
        "sereh": "serai", "serei": "serai",
        "daun bawang": "daun bawang",
        "telor": "telur",
        "toge": "tauge",
        "jeroan": "jeroan",
        "lemak": "minyak goreng",
    }

    # ------------------------------------------------------------------
    # TEKNIK MEMASAK  (untuk filter recipes.kategori / langkah_langkah)
    # ------------------------------------------------------------------
    COOKING_METHODS: List[str] = [
        "goreng", "digoreng", "menggoreng", "tumis", "ditumis", "menumis",
        "rebus", "direbus", "merebus", "kukus", "dikukus", "mengukus",
        "panggang", "dipanggang", "memanggang", "bakar", "dibakar", "membakar",
        "sate", "pepes", "opor", "semur", "rendang", "soto", "gulai",
        "kari", "sayur", "sup", "sop", "bening",
        "sangrai", "capcay", "stir fry",
        "steam", "bake",
    ]

    # ------------------------------------------------------------------
    # KONDISI KESEHATAN  (sesuai health_conditions.nama di database)
    # ------------------------------------------------------------------
    HEALTH_CONDITIONS: Dict[str, str] = {
        # kata kunci → nama standar (health_conditions.nama)
        "diabetes": "Diabetes", "diabetik": "Diabetes",
        "diabetus": "Diabetes", "kencing manis": "Diabetes",
        "gula darah": "Diabetes", "gula darah tinggi": "Diabetes",

        "kolesterol": "Kolesterol Tinggi", "kolestrol": "Kolesterol Tinggi",
        "kolesterol tinggi": "Kolesterol Tinggi",
        "lemak darah": "Kolesterol Tinggi",

        "hipertensi": "Hipertensi", "tekanan darah tinggi": "Hipertensi",
        "darah tinggi": "Hipertensi",

        "asam urat": "Asam Urat", "asam_urat": "Asam Urat",
        "gout": "Asam Urat",

        "maag": "Maag", "mag": "Maag", "gastritis": "Maag",
        "sakit lambung": "Maag", "tukak lambung": "Maag",
        "lambung": "Maag",

        "alergi susu": "Alergi Susu", "dairy free": "Alergi Susu",
        "intoleransi laktosa": "Alergi Susu", "laktosa": "Alergi Susu",

        "alergi gluten": "Alergi Gluten", "gluten free": "Alergi Gluten",
        "celiac": "Alergi Gluten",

        "obesitas": "Obesitas", "kegemukan": "Obesitas",
        "diet": "Diet Sehat", "diet sehat": "Diet Sehat",
        "diet ketat": "Diet Sehat",

        "vegetarian": "Vegetarian",
        "vegan": "Vegan",

        "jantung": "Penyakit Jantung", "sakit jantung": "Penyakit Jantung",
        "penyakit jantung": "Penyakit Jantung",

        "anemia": "Anemia", "kurang darah": "Anemia",
    }

    # Bahan yang WAJIB dihindari per kondisi (severity=hindari)
    CONDITION_RESTRICTIONS: Dict[str, List[str]] = {
        "Diabetes": ["gula", "gula pasir", "gula merah", "nasi putih", "roti", "mie", "tepung", "kecap manis"],
        "Kolesterol Tinggi": ["santan", "santan kental", "minyak goreng", "mentega", "butter", "margarin", "jeroan"],
        "Hipertensi": ["garam", "kecap asin", "saus tiram", "penyedap rasa"],
        "Asam Urat": ["jeroan", "udang", "kepiting", "kerang", "sarden", "kacang panjang", "bayam"],
        "Maag": ["cabai", "cabe", "cabai rawit", "cabe rawit", "asam jawa", "cuka"],
        "Alergi Susu": ["susu", "keju", "butter", "mentega", "yogurt"],
        "Alergi Gluten": ["tepung terigu", "roti tawar", "pasta", "spaghetti", "mie"],
        "Obesitas": ["minyak goreng", "santan", "gula", "nasi"],
        "Vegan": ["daging sapi", "daging ayam", "ayam", "ikan", "udang", "telur", "susu", "keju"],
        "Vegetarian": ["daging sapi", "daging ayam", "ayam", "ikan", "udang"],
        "Penyakit Jantung": ["santan", "minyak goreng", "jeroan", "daging sapi", "garam"],
        "Anemia": [],
        "Diet Sehat": ["minyak goreng", "gula", "garam", "santan"],
    }

    # ------------------------------------------------------------------
    # PREFERENSI RASA
    # ------------------------------------------------------------------
    TASTE_KEYWORDS: Dict[str, str] = {
        "pedas": "pedas", "pedes": "pedas", "panas": "pedas",
        "manis": "manis",
        "asin": "asin", "gurih": "gurih",
        "asam": "asam", "segar": "segar",
        "pahit": "pahit",
        "tidak pedas": "tidak_pedas", "ga pedas": "tidak_pedas",
        "gak pedas": "tidak_pedas",
        "tidak manis": "tidak_manis",
    }

    # ------------------------------------------------------------------
    # REGION / DAERAH  (sesuai recipes.region)
    # ------------------------------------------------------------------
    REGION_KEYWORDS: Dict[str, str] = {
        "jawa": "Jawa", "javanese": "Jawa",
        "sunda": "Sunda", "sundanese": "Sunda",
        "padang": "Padang", "minang": "Padang",
        "betawi": "Betawi",
        "bali": "Bali", "balinese": "Bali",
        "manado": "Manado", "minahasa": "Manado",
        "aceh": "Aceh",
        "melayu": "Melayu",
        "kalimantan": "Kalimantan",
        "sulawesi": "Sulawesi",
        "chinese": "Chinese", "cina": "Chinese", "tionghoa": "Chinese",
        "western": "Western", "eropa": "Western", "barat": "Western",
        "italia": "Italia", "italian": "Italia",
        "jepang": "Jepang", "japanese": "Jepang",
        "korea": "Korea", "korean": "Korea",
        "india": "India", "indian": "India",
        "timur tengah": "Timur Tengah",
    }

    # ------------------------------------------------------------------
    # POLA NEGASI (bahan yang harus dihindari)
    # ------------------------------------------------------------------
    AVOID_PATTERNS = [
        r"tanpa\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
        r"tidak\s+pakai\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
        r"tidak\s+boleh\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
        r"ga\s+boleh\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
        r"gak\s+boleh\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
        r"hindari\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
        r"ga\s+bisa\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
        r"alergi\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
        r"tidak\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all(self, text: str) -> Dict:
        """
        Ekstrak semua entities dari teks.
        Output diselaraskan dengan kolom database:
        {
            ingredients: {main: [...], avoid: [...]},
            cooking_methods: [...],
            health_conditions: [...],         # nilai = health_conditions.nama
            taste_preferences: [...],
            time_constraint: int | None,       # dalam menit → recipes.waktu_masak
            region: str | None,               # → recipes.region
        }
        """
        text_lower = text.lower()

        avoid_explicit = self._extract_avoid(text_lower)
        main_ingredients = self._extract_main_ingredients(text_lower, avoid_explicit)
        cooking_methods = self._extract_cooking_methods(text_lower)
        health_conditions = self._extract_health_conditions(text_lower)
        taste_preferences = self._extract_taste_preferences(text_lower)
        time_constraint = self._extract_time_constraint(text_lower)
        region = self._extract_region(text_lower)

        # Gabungkan avoid dari kondisi kesehatan
        condition_avoid: List[str] = []
        for cond_name in health_conditions:
            condition_avoid.extend(self.CONDITION_RESTRICTIONS.get(cond_name, []))

        all_avoid = list(set(avoid_explicit + condition_avoid))

        return {
            "ingredients": {
                "main": main_ingredients,
                "avoid": all_avoid,
            },
            "cooking_methods": cooking_methods,
            "health_conditions": health_conditions,
            "taste_preferences": taste_preferences,
            "time_constraint": time_constraint,
            "region": region,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_avoid(self, text: str) -> List[str]:
        """Ekstrak bahan yang harus dihindari dari pola negasi."""
        avoided = []
        for pattern in self.AVOID_PATTERNS:
            for match in re.finditer(pattern, text):
                phrase = match.group(1).strip()
                # Cek apakah phrase cocok dengan ingredient yang dikenal
                normalized = self.INGREDIENT_SYNONYMS.get(phrase, phrase)
                if normalized in self.INGREDIENTS:
                    avoided.append(normalized)
                else:
                    # Cek n-gram dalam phrase
                    for ing in sorted(self.INGREDIENTS.keys(), key=len, reverse=True):
                        if ing in phrase:
                            avoided.append(ing)
                            break
        return list(set(avoided))

    def _extract_main_ingredients(self, text: str, avoid: List[str]) -> List[str]:
        """Ekstrak bahan utama menggunakan n-gram matching."""
        words = text.split()
        found: List[str] = []
        used_positions = set()

        # Urutkan dari n-gram terpanjang → terpendek agar tidak salah potong
        sorted_ingredients = sorted(self.INGREDIENTS.keys(), key=len, reverse=True)

        for ing in sorted_ingredients:
            ing_words = ing.split()
            n = len(ing_words)
            for i in range(len(words) - n + 1):
                if any(j in used_positions for j in range(i, i + n)):
                    continue
                if words[i:i + n] == ing_words:
                    canonical = self.INGREDIENT_SYNONYMS.get(ing, ing)
                    if canonical not in avoid:
                        found.append(canonical)
                    for j in range(i, i + n):
                        used_positions.add(j)

        return list(dict.fromkeys(found))  # deduplicate, preserve order

    def _extract_cooking_methods(self, text: str) -> List[str]:
        """Ekstrak teknik memasak."""
        found = []
        for method in self.COOKING_METHODS:
            if method in text:
                found.append(method)
        return list(set(found))

    def _extract_health_conditions(self, text: str) -> List[str]:
        """
        Ekstrak kondisi kesehatan.
        Mengembalikan nama standar sesuai health_conditions.nama.
        """
        found = []
        # Urutkan panjang keyword dari panjang ke pendek (hindari false match)
        sorted_keys = sorted(self.HEALTH_CONDITIONS.keys(), key=len, reverse=True)
        for keyword in sorted_keys:
            if keyword in text:
                std_name = self.HEALTH_CONDITIONS[keyword]
                if std_name not in found:
                    found.append(std_name)
        return found

    def _extract_taste_preferences(self, text: str) -> List[str]:
        """Ekstrak preferensi rasa."""
        found = []
        sorted_keys = sorted(self.TASTE_KEYWORDS.keys(), key=len, reverse=True)
        for keyword in sorted_keys:
            if keyword in text:
                taste = self.TASTE_KEYWORDS[keyword]
                if taste not in found:
                    found.append(taste)
        return found

    def _extract_time_constraint(self, text: str) -> Optional[int]:
        """
        Ekstrak waktu memasak dalam menit (sesuai recipes.waktu_masak).
        Returns: jumlah menit (int) atau None
        """
        # Pola eksplisit: "30 menit", "1 jam"
        m = re.search(r"(\d+)\s*menit", text)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)\s*jam", text)
        if m:
            return int(m.group(1)) * 60

        # Kata kunci waktu relatif
        if any(w in text for w in ["cepat", "cepet", "kilat", "ekspres"]):
            return 30   # ≤ 30 menit
        if any(w in text for w in ["simpel", "simple", "gampang", "mudah"]):
            return 45   # ≤ 45 menit
        return None

    def _extract_region(self, text: str) -> Optional[str]:
        """Ekstrak region masakan (sesuai recipes.region)."""
        sorted_keys = sorted(self.REGION_KEYWORDS.keys(), key=len, reverse=True)
        for keyword in sorted_keys:
            if keyword in text:
                return self.REGION_KEYWORDS[keyword]
        return None


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    ner = NERExtractor()
    cases = [
        "mau masak ayam goreng yang crispy tapi tanpa tepung",
        "pengen bikin pasta carbonara tapi dairy free karena alergi susu",
        "aku diabetes jadi ga boleh makan yang manis manis",
        "mau yang pedas gurih, direbus aja biar sehat",
        "cariin resep ikan bakar yang cepat dan gampang",
        "kolesterol tinggi jadi ga bisa santan dan minyak goreng",
        "masakan padang yang tidak terlalu pedas",
        "mau masak soto ayam jawa yang enak",
    ]
    for text in cases:
        print(f"Input: {text}")
        result = ner.extract_all(text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("-" * 80)