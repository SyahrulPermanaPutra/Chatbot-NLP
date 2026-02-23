# src/preprocessor.py
# Text preprocessing dengan normalisasi kata informal Indonesia

import re
import json
from typing import Dict, List


class TextPreprocessor:
    """
    Kelas untuk preprocessing teks input user
    - Normalisasi kata informal ke formal
    - Koreksi typo umum
    - Cleaning karakter khusus
    """

    # -------------------------------------------------------
    # Kamus normalisasi kata informal (tidak perlu file eksternal)
    # -------------------------------------------------------
    INFORMAL_MAP: Dict[str, str] = {
        # Kata ganti / ekspresi umum
        "gw": "saya", "gue": "saya", "aku": "saya", "w": "saya",
        "lo": "kamu", "lu": "kamu",
        "mau": "ingin", "pengen": "ingin", "kepingin": "ingin", "pingin": "ingin",
        "ga": "tidak", "gak": "tidak", "ngga": "tidak", "nggak": "tidak",
        "gapapa": "tidak apa", "gpp": "tidak apa",
        "udah": "sudah", "udh": "sudah",
        "gimana": "bagaimana", "gmn": "bagaimana",
        "kayak": "seperti", "kyk": "seperti",
        "aja": "saja", "doank": "saja", "doang": "saja",
        "banget": "sangat", "bgt": "sangat", "bget": "sangat",
        "yg": "yang", "dgn": "dengan", "sm": "sama",
        "tp": "tapi", "tapi": "tapi",
        "krn": "karena", "karna": "karena",
        "utk": "untuk", "buat": "untuk",
        "deh": "", "dong": "", "sih": "", "nih": "",
        "gt": "itu", "gitu": "itu", "gini": "ini",
        "ok": "oke", "oke": "oke",
        "hbs": "habis", "abis": "habis",
        "sdh": "sudah", "blm": "belum", "blum": "belum",
        "jd": "jadi", "jadi": "jadi",
        "br": "baru",
        "klo": "kalau", "kalo": "kalau", "kl": "kalau",
        "bs": "bisa", "bsa": "bisa",

        # Kata memasak informal
        "bikin": "membuat", "bikinin": "buatkan",
        "masakin": "masakan", "ngegoreng": "menggoreng",
        "ngerebus": "merebus", "ngetumis": "menumis",
        "digoreng": "goreng", "direbus": "rebus",
        "dipanggang": "panggang", "ditumis": "tumis",
        "dipake": "dipakai", "pake": "pakai",
        "nyari": "mencari", "cariin": "carikan",

        # Kata bahan/makanan informal
        "aym": "ayam", "ayem": "ayam",
        "ikannya": "ikan",
        "dagingnya": "daging",
        "sayurnya": "sayuran",
        "cepet": "cepat",
        "simpel": "simple",
        "gampang": "mudah",
    }

    TYPO_MAP: Dict[str, str] = {
        # Bahan protein
        "ayma": "ayam", "aymm": "ayam",
        "iakn": "ikan", "ikna": "ikan",
        "udaang": "udang", "udangg": "udang",
        "tmpe": "tempe", "temep": "tempe",
        "thau": "tahu", "taahu": "tahu",
        "telour": "telur", "telor": "telur",
        "dagng": "daging", "dagin": "daging",
        "sapy": "sapi",
        "kambng": "kambing",

        # Bahan sayuran
        "bayem": "bayam", "bayamm": "bayam",
        "wortle": "wortel", "wortel": "wortel",
        "kentag": "kentang", "kentangg": "kentang",
        "tomat": "tomat",
        "terong": "terong",
        "kangkong": "kangkung",

        # Bumbu
        "bawng": "bawang", "bawnag": "bawang",
        "jae": "jahe", "jaeh": "jahe",
        "kunir": "kunyit", "kunirt": "kunyit",
        "serei": "serai", "sereh": "serai",

        # Teknik memasak
        "goreing": "goreng", "goreg": "goreng",
        "tumsi": "tumis",
        "reubs": "rebus",
        "pangagng": "panggang",
        "kuksu": "kukus",

        # Kondisi kesehatan
        "diabets": "diabetes", "diabetus": "diabetes",
        "kolestreol": "kolesterol", "kolestrol": "kolesterol",
        "hipertesi": "hipertensi",
        "asam urat": "asam_urat",
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> Dict:
        """
        Main preprocessing function.
        Returns dict: {original, normalized, negations}
        """
        return {
            "original": text,
            "normalized": self.normalize_text(text),
            "negations": self.extract_negations(text),
        }

    def normalize_text(self, text: str) -> str:
        """Pipeline normalisasi lengkap."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)   # hapus tanda baca
        text = self._normalize_informal_words(text)
        text = self._fix_common_typos(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_negations(self, text: str) -> List[str]:
        """Ekstrak pola negasi dalam teks."""
        patterns = [
            r"tidak\s+\w+",
            r"tanpa\s+\w+",
            r"gak\s+\w+",
            r"ga\s+\w+",
            r"jangan\s+\w+",
            r"hindari\s+\w+",
            r"ga\s+boleh\s+\w+",
            r"gak\s+bisa\s+\w+",
            r"tidak\s+boleh\s+\w+",
            r"tidak\s+bisa\s+\w+",
            r"alergi\s+\w+",
        ]
        found = []
        for pat in patterns:
            found.extend(re.findall(pat, text, re.IGNORECASE))
        return found

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_informal_words(self, text: str) -> str:
        words = text.split()
        result = []
        for w in words:
            replacement = self.INFORMAL_MAP.get(w, w)
            if replacement:           # buang kata kosong (misal "deh"→"")
                result.append(replacement)
        return " ".join(result)

    def _fix_common_typos(self, text: str) -> str:
        words = text.split()
        return " ".join(self.TYPO_MAP.get(w, w) for w in words)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pp = TextPreprocessor()
    cases = [
        "gw pengen masak aym gorng yg krispy bgt tapi gak pake tepung",
        "mau bikin pasta carbonara tp dairy free gimana caranya",
        "aku diabets jd ga boleh makan yg manis manis",
        "cariin resep sayur asem donk yg gampang",
    ]
    for c in cases:
        r = pp.preprocess(c)
        print(f"Original : {r['original']}")
        print(f"Normalized: {r['normalized']}")
        print(f"Negations : {r['negations']}")
        print("-" * 60)
