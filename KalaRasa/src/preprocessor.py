# src/preprocessor.py  (v2 – Fase 1) - FIXED VERSION
"""
Text preprocessing dengan:
  - Kamus dinamic dari file JSON (bisa diedit non-developer)
  - Synonym expansion untuk NER
  - Fallback ke konstanta hardcoded jika file tidak ada
"""

import re
import json
import os
import glob
from typing import Dict, List, Optional


class TextPreprocessor:
    """
    Kelas preprocessing teks input user.

    Perubahan v (Fase 1):
    - INFORMAL_MAP dan TYPO_MAP bisa diload dari file JSON eksternal
    - Synonym expansion untuk normalisasi sebelum NER
    - Fallback otomatis ke konstanta hardcoded jika file tidak ada
    """

    # ── Fallback hardcoded (dipakai jika file JSON tidak ada) ────────

    _BUILTIN_INFORMAL_MAP: Dict[str, str] = {
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
        "tp": "tapi",
        "krn": "karena", "karna": "karena",
        "utk": "untuk", "buat": "untuk",
        "deh": "", "dong": "", "sih": "", "nih": "",
        "gt": "itu", "gitu": "itu", "gini": "ini",
        "ok": "oke", "oke": "oke",
        "hbs": "habis", "abis": "habis",
        "sdh": "sudah", "blm": "belum", "blum": "belum",
        "jd": "jadi",
        "br": "baru",
        "klo": "kalau", "kalo": "kalau", "kl": "kalau",
        "bs": "bisa", "bsa": "bisa",
        "bikin": "membuat", "bikinin": "buatkan",
        "masakin": "masakan", "ngegoreng": "menggoreng",
        "ngerebus": "merebus", "ngetumis": "menumis",
        "digoreng": "goreng", "direbus": "rebus",
        "dipanggang": "panggang", "ditumis": "tumis",
        "dipake": "dipakai", "pake": "pakai",
        "nyari": "mencari", "cariin": "carikan",
        "aym": "ayam", "ayem": "ayam",
        "ikannya": "ikan", "dagingnya": "daging", "sayurnya": "sayuran",
        "cepet": "cepat", "simpel": "simple", "gampang": "mudah",
    }

    _BUILTIN_TYPO_MAP: Dict[str, str] = {
        "ayma": "ayam", "aymm": "ayam",
        "iakn": "ikan", "ikna": "ikan",
        "udaang": "udang", "udangg": "udang",
        "tmpe": "tempe", "temep": "tempe",
        "thau": "tahu", "taahu": "tahu",
        "telour": "telur", "telor": "telur",
        "dagng": "daging", "dagin": "daging",
        "sapy": "sapi", "kambng": "kambing",
        "bayem": "bayam", "bayamm": "bayam",
        "wortle": "wortel",
        "kentag": "kentang", "kentangg": "kentang",
        "kangkong": "kangkung",
        "bawng": "bawang", "bawnag": "bawang",
        "jae": "jahe", "jaeh": "jahe",
        "kunir": "kunyit", "kunirt": "kunyit",
        "serei": "serai", "sereh": "serai",
        "goreing": "goreng", "goreg": "goreng",
        "tumsi": "tumis",
        "reubs": "rebus",
        "pangagng": "panggang",
        "kuksu": "kukus",
        "diabets": "diabetes", "diabetus": "diabetes",
        "kolestreol": "kolesterol", "kolestrol": "kolesterol",
        "hipertesi": "hipertensi",
        "asam urat": "asam_urat",
    }

    # ────────────────────────────────────────────────────────────────────

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: direktori root tempat data/informal_map.json dan
                      data/synonyms/*.json berada. Default: 'data'
        """
        self.data_dir = data_dir
        self._json_loaded = False

        # Load kamus dari file JSON; fallback ke builtin jika gagal
        self.informal_map: Dict[str, str] = self._load_informal_map()
        self.typo_map:     Dict[str, str] = self._load_typo_map()  # Added this line

        # Synonym map: { canonical_form: [sinonim, ...] }
        self.synonym_map:  Dict[str, List[str]] = self._load_synonyms()

        # Reverse synonym: { sinonim → canonical } untuk normalisasi
        self._reverse_synonym: Dict[str, str] = self._build_reverse_synonym()

        print(f"  ✓ Preprocessor v2 ready "
              f"(informal={len(self.informal_map)}, "
              f"typo={len(self.typo_map)}, "
              f"synonyms={len(self._reverse_synonym)}, "
              f"source={'json' if self._json_loaded else 'builtin'})")

    # ── Loader ─────────────────────────────────────────────────────────

    def _load_informal_map(self) -> Dict[str, str]:
        """Load informal_map.json; fallback ke builtin."""
        path = os.path.join(self.data_dir, "informal_map.json")
        self._json_loaded = False
        
        try:
            if not os.path.exists(path):
                print(f"  ⚠ {path} tidak ditemukan, menggunakan builtin informal_map")
                return self._BUILTIN_INFORMAL_MAP.copy()
                
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            
            # Handle different JSON structures
            result = {}
            
            if isinstance(raw, dict):
                for k, v in raw.items():
                    # Skip metadata keys
                    if k.startswith("_"):
                        continue
                    
                    # Handle both string values and nested objects
                    if isinstance(v, str):
                        result[k] = v
                    elif isinstance(v, dict) and "formal" in v:
                        result[k] = v["formal"]
                    elif isinstance(v, list) and len(v) > 0:
                        result[k] = v[0]  # Take first item as formal version
            
            self._json_loaded = len(result) > 0
            if self._json_loaded:
                print(f"  ✓ Loaded {len(result)} entries from {path}")
                return result
            else:
                print(f"  ⚠ No valid entries in {path}, using builtin")
                return self._BUILTIN_INFORMAL_MAP.copy()
                
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON decode error in {path}: {e}, menggunakan builtin")
            return self._BUILTIN_INFORMAL_MAP.copy()
        except Exception as e:
            print(f"  ⚠ Gagal load {path}: {e}, menggunakan builtin")
            return self._BUILTIN_INFORMAL_MAP.copy()

    def _load_typo_map(self) -> Dict[str, str]:
        """Load typo_map.json if exists, otherwise use builtin."""
        path = os.path.join(self.data_dir, "typo_map.json")
        
        try:
            if not os.path.exists(path):
                return self._BUILTIN_TYPO_MAP.copy()
                
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            
            result = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if not k.startswith("_") and isinstance(v, str):
                        result[k] = v
            
            if result:
                print(f"  ✓ Loaded {len(result)} entries from {path}")
                return result
            else:
                return self._BUILTIN_TYPO_MAP.copy()
                
        except Exception as e:
            print(f"  ⚠ Gagal load typo map: {e}, menggunakan builtin")
            return self._BUILTIN_TYPO_MAP.copy()

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load semua *.json di data/synonyms/."""
        synonyms: Dict[str, List[str]] = {}
        synonym_dir = os.path.join(self.data_dir, "synonyms")

        if not os.path.isdir(synonym_dir):
            print(f"  ⚠ Synonym directory not found: {synonym_dir}")
            return synonyms

        loaded_files = 0
        for filepath in glob.glob(os.path.join(synonym_dir, "*.json")):
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    for canonical, variants in data.items():
                        if canonical.startswith("_"):
                            continue
                        if isinstance(variants, list):
                            existing = synonyms.get(canonical, [])
                            # Filter and add valid variants
                            valid_variants = [v for v in variants if isinstance(v, str)]
                            synonyms[canonical] = list(set(existing + valid_variants))
                        elif isinstance(variants, str):
                            # Handle single string variant
                            existing = synonyms.get(canonical, [])
                            synonyms[canonical] = list(set(existing + [variants]))
                    
                    loaded_files += 1
            except Exception as e:
                print(f"  ⚠ Gagal load {filepath}: {e}")

        if loaded_files > 0:
            total_variants = sum(len(v) for v in synonyms.values())
            print(f"  ✓ Loaded {loaded_files} synonym files with {total_variants} variants")
        
        return synonyms

    def _build_reverse_synonym(self) -> Dict[str, str]:
        """Bangun reverse lookup: sinonim → canonical."""
        reverse: Dict[str, str] = {}
        for canonical, variants in self.synonym_map.items():
            # Add canonical to canonical mapping
            reverse[canonical.lower()] = canonical
            # Add all variants
            for variant in variants:
                if isinstance(variant, str):
                    reverse[variant.lower()] = canonical
        return reverse

    # ── Public API ──────────────────────────────────────────────────────

    def preprocess(self, text: str) -> Dict:
        """
        Main preprocessing function.
        Returns dict: {original, normalized, negations, expanded_terms}
        """
        normalized = self.normalize_text(text)
        return {
            "original":       text,
            "normalized":     normalized,
            "negations":      self.extract_negations(text),
            "expanded_terms": self._expand_synonyms(normalized),
        }

    def normalize_text(self, text: str) -> str:
        """Pipeline normalisasi lengkap."""
        if not text or not isinstance(text, str):
            return ""
            
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = self._normalize_informal_words(text)
        text = self._fix_common_typos(text)
        text = self._apply_reverse_synonyms(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_negations(self, text: str) -> List[str]:
        """Ekstrak pola negasi dalam teks."""
        patterns = [
            r"tidak\s+\w+", r"tanpa\s+\w+", r"gak\s+\w+",
            r"ga\s+\w+", r"jangan\s+\w+", r"hindari\s+\w+",
            r"ga\s+boleh\s+\w+", r"gak\s+bisa\s+\w+",
            r"tidak\s+boleh\s+\w+", r"tidak\s+bisa\s+\w+",
            r"alergi\s+\w+",
        ]
        found = []
        for pat in patterns:
            found.extend(re.findall(pat, text, re.IGNORECASE))
        return found

    def reload_dictionaries(self):
        """
        Reload semua kamus dari file tanpa restart service.
        Dipanggil setelah file JSON diupdate oleh tim konten.
        """
        self.informal_map      = self._load_informal_map()
        self.typo_map          = self._load_typo_map()
        self.synonym_map       = self._load_synonyms()
        self._reverse_synonym  = self._build_reverse_synonym()
        print("  ✓ Dictionaries reloaded from disk")

    # ── Private helpers ─────────────────────────────────────────────────

    def _normalize_informal_words(self, text: str) -> str:
        if not text:
            return text
        words = text.split()
        result = []
        for w in words:
            replacement = self.informal_map.get(w, w)
            if replacement:  # Only append if replacement is not empty string
                result.append(replacement)
        return " ".join(result)

    def _fix_common_typos(self, text: str) -> str:
        if not text:
            return text
        words = text.split()
        return " ".join(self.typo_map.get(w, w) for w in words)

    def _apply_reverse_synonyms(self, text: str) -> str:
        """Normalisasi sinonim ke canonical form (contoh: 'chicken' → 'ayam')."""
        if not text:
            return text
        words = text.split()
        return " ".join(self._reverse_synonym.get(w, w) for w in words)

    def _expand_synonyms(self, normalized_text: str) -> Dict[str, str]:
        """
        Cari canonical terms yang ditemukan dalam teks.
        Returns: { canonical: found_word }
        Berguna untuk NER downstream.
        """
        found: Dict[str, str] = {}
        if not normalized_text:
            return found
            
        words = set(normalized_text.split())
        for canonical, variants in self.synonym_map.items():
            if canonical in words:
                found[canonical] = canonical
            else:
                for v in variants:
                    if v.lower() in words:
                        found[canonical] = v
                        break
        return found


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with absolute path to help debug
    import sys
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Try multiple possible data directories
    possible_paths = [
        "data",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
        os.path.join(os.getcwd(), "data")
    ]
    
    for data_path in possible_paths:
        abs_path = os.path.abspath(data_path)
        print(f"\nTrying data_dir: {abs_path}")
        print(f"  Exists: {os.path.exists(abs_path)}")
        
        if os.path.exists(abs_path):
            informal_file = os.path.join(abs_path, "informal_map.json")
            print(f"  informal_map.json exists: {os.path.exists(informal_file)}")
            
            if os.path.exists(informal_file):
                try:
                    with open(informal_file, encoding="utf-8") as f:
                        content = f.read()[:200]  # First 200 chars
                    print(f"  File content preview: {content}...")
                except Exception as e:
                    print(f"  Error reading file: {e}")
    
    print("\n" + "="*60)
    
    # Test the preprocessor
    pp = TextPreprocessor(data_dir="data")  # Adjust this path as needed
    
    cases = [
        "gw pengen masak aym gorng yg krispy bgt tapi gak pake tepung",
        "mau bikin pasta carbonara tp dairy free gimana caranya",
        "aku diabets jd ga boleh makan yg manis manis",
        "cariin resep chicken goreng donk yg gampang",
    ]
    
    for c in cases:
        r = pp.preprocess(c)
        print(f"\nOriginal     : {r['original']}")
        print(f"Normalized   : {r['normalized']}")
        print(f"Negations    : {r['negations']}")
        print(f"Expanded     : {r['expanded_terms']}")
        print("-" * 60)