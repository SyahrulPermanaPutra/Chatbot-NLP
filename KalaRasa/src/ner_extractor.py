# src/ner_extractor.py (FIXED VERSION)

import re
import json
import os
from typing import Dict, List, Optional, Set, Any, Union


class NERExtractor:
    """
    Rule-based NER extractor yang entitiesnya selaras dengan database kala_rasa_jtv.
    Menggunakan data dari file ner.json eksternal dengan struktur yang diberikan.
    """

    # Path default ke file ner.json
    DEFAULT_NER_JSON_PATH = "C:/Users/MyBook Hype AMD/Chatbot-NLP/KalaRasa/data/ner.json"

    def __init__(self, ner_json_path: Optional[str] = None):
        """
        Initialize NER extractor dengan data dari file JSON.
        
        Args:
            ner_json_path: Path ke file ner.json. Jika None, menggunakan DEFAULT_NER_JSON_PATH
        """
        self.ner_json_path = ner_json_path or self.DEFAULT_NER_JSON_PATH
        
        # Inisialisasi dictionary kosong
        self.INGREDIENTS: Dict[str, str] = {}
        self.INGREDIENT_SYNONYMS: Dict[str, str] = {}
        self.COOKING_METHODS: List[str] = []
        self.HEALTH_CONDITIONS: Dict[str, str] = {}
        self.CONDITION_RESTRICTIONS: Dict[str, List[str]] = {}
        self.TASTE_KEYWORDS: Dict[str, str] = {}
        self.REGION_KEYWORDS: Dict[str, str] = {}
        
        # Load data dari JSON
        self._load_ner_data()
        
        # Prepare lookup structures untuk performance
        self._prepare_lookups()

    def _load_ner_data(self):
        """Load data NER dari file JSON dengan struktur yang diberikan."""
        try:
            if not os.path.isfile(self.ner_json_path):
                print(f"⚠ Warning: File ner.json tidak ditemukan di {self.ner_json_path}")
                print("  Menggunakan data default (fallback)")
                self._load_default_data()
                return

            with open(self.ner_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n✓ Loading NER data from: {self.ner_json_path}")
            
            # ------------------------------------------------------------
            # Load ingredients (LIST format)
            # ------------------------------------------------------------
            ingredients_list = data.get("ingredients", [])
            if isinstance(ingredients_list, list):
                for ingredient in ingredients_list:
                    if isinstance(ingredient, str):
                        # Default category untuk ingredient adalah "umum"
                        self.INGREDIENTS[ingredient.lower()] = "umum"
            print(f"  - Ingredients: {len(self.INGREDIENTS)} items")
            
            # ------------------------------------------------------------
            # Load ingredient synonyms (LIST format with ":" separator)
            # ------------------------------------------------------------
            synonyms_list = data.get("ingredient_synonyms", [])
            if isinstance(synonyms_list, list):
                for item in synonyms_list:
                    if isinstance(item, str) and ":" in item:
                        parts = item.split(":", 1)
                        informal = parts[0].strip().lower()
                        formal = parts[1].strip().lower()
                        if informal and formal:
                            self.INGREDIENT_SYNONYMS[informal] = formal
            print(f"  - Synonyms: {len(self.INGREDIENT_SYNONYMS)} items")
            
            # ------------------------------------------------------------
            # Load cooking methods (LIST format)
            # ------------------------------------------------------------
            methods_list = data.get("cooking_methods", [])
            if isinstance(methods_list, list):
                self.COOKING_METHODS = [m.lower() for m in methods_list if isinstance(m, str)]
            print(f"  - Cooking methods: {len(self.COOKING_METHODS)} items")
            
            # ------------------------------------------------------------
            # Load health conditions (LIST format with ":" separator)
            # ------------------------------------------------------------
            health_list = data.get("health_conditions", [])
            if isinstance(health_list, list):
                for item in health_list:
                    if isinstance(item, str) and ":" in item:
                        parts = item.split(":", 1)
                        keyword = parts[0].strip().lower()
                        condition = parts[1].strip()
                        if keyword and condition:
                            self.HEALTH_CONDITIONS[keyword] = condition
            print(f"  - Health conditions: {len(self.HEALTH_CONDITIONS)} items")
            
            # ------------------------------------------------------------
            # Load condition restrictions (LIST format with ":" separator)
            # ------------------------------------------------------------
            restrictions_list = data.get("condition_restrictions", [])
            if isinstance(restrictions_list, list):
                for item in restrictions_list:
                    if isinstance(item, str) and ":" in item:
                        parts = item.split(":", 1)
                        condition = parts[0].strip()
                        ingredients_str = parts[1].strip()
                        if condition and ingredients_str:
                            # Split by comma and clean up
                            ingredients = [ing.strip().lower() for ing in ingredients_str.split(",") if ing.strip()]
                            self.CONDITION_RESTRICTIONS[condition] = ingredients
            print(f"  - Condition restrictions: {len(self.CONDITION_RESTRICTIONS)} conditions")
            
            # ------------------------------------------------------------
            # Load taste keywords (LIST format with ":" separator)
            # ------------------------------------------------------------
            taste_list = data.get("taste_keywords", [])
            if isinstance(taste_list, list):
                for item in taste_list:
                    if isinstance(item, str) and ":" in item:
                        parts = item.split(":", 1)
                        keyword = parts[0].strip().lower()
                        taste = parts[1].strip().lower()
                        if keyword and taste:
                            self.TASTE_KEYWORDS[keyword] = taste
            print(f"  - Taste keywords: {len(self.TASTE_KEYWORDS)} items")
            
            # ------------------------------------------------------------
            # Load region keywords (LIST format with ":" separator)
            # ------------------------------------------------------------
            region_list = data.get("region_keywords", [])
            if isinstance(region_list, list):
                for item in region_list:
                    if isinstance(item, str) and ":" in item:
                        parts = item.split(":", 1)
                        keyword = parts[0].strip().lower()
                        region = parts[1].strip()
                        if keyword and region:
                            self.REGION_KEYWORDS[keyword] = region
            print(f"  - Region keywords: {len(self.REGION_KEYWORDS)} items")
            
        except Exception as e:
            print(f"⚠ Error loading ner.json: {e}")
            print("  Menggunakan data default (fallback)")
            self._load_default_data()

    def _load_default_data(self):
        """Load data default minimal (fallback jika JSON error)."""
        self.INGREDIENTS = {
            "ayam": "protein", "ikan": "protein", "tempe": "protein",
            "tahu": "protein", "telur": "protein",
            "bayam": "sayuran", "kangkung": "sayuran", "wortel": "sayuran",
            "nasi": "karbohidrat", "mie": "karbohidrat", "kentang": "karbohidrat",
            "bawang": "bumbu", "cabai": "bumbu",
            "minyak": "lemak", "santan": "lemak",
            "garam": "penyedap", "gula": "penyedap",
        }
        
        self.COOKING_METHODS = ["goreng", "rebus", "kukus", "tumis", "bakar"]
        
        self.HEALTH_CONDITIONS = {
            "diabetes": "Diabetes", "kolesterol": "Kolesterol Tinggi",
            "darah tinggi": "Hipertensi", "asam urat": "Asam Urat",
        }
        
        self.TASTE_KEYWORDS = {"pedas": "pedas", "manis": "manis", "asin": "asin"}
        self.REGION_KEYWORDS = {"padang": "Padang", "jawa": "Jawa", "sunda": "Sunda"}

    def _prepare_lookups(self):
        """Prepare lookup structures untuk performance."""
        # Sort ingredients by length (longest first) for matching
        self._sorted_ingredients = sorted(self.INGREDIENTS.keys(), key=len, reverse=True)
        
        # Sort health condition keywords by length
        self._sorted_health_keys = sorted(self.HEALTH_CONDITIONS.keys(), key=len, reverse=True)
        
        # Sort taste keywords by length
        self._sorted_taste_keys = sorted(self.TASTE_KEYWORDS.keys(), key=len, reverse=True)
        
        # Sort region keywords by length
        self._sorted_region_keys = sorted(self.REGION_KEYWORDS.keys(), key=len, reverse=True)
        
        # Define avoid patterns
        self.AVOID_PATTERNS = [
            r"tanpa\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
            r"tidak\s+pakai\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
            r"tidak\s+boleh\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
            r"hindari\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
            r"alergi\s+([\w\s]+?)(?:\s+dan|\s+atau|\s*,|\s*$)",
        ]
        
        self._compiled_avoid_patterns = [re.compile(p, re.IGNORECASE) for p in self.AVOID_PATTERNS]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all(self, text: str) -> Dict:
        """
        Ekstrak semua entities dari teks.
        Output diselaraskan dengan kolom database.
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
        avoided = set()
        
        for pattern in self._compiled_avoid_patterns:
            for match in pattern.finditer(text):
                phrase = match.group(1).strip().lower()
                if not phrase:
                    continue
                    
                # Cek apakah phrase cocok dengan ingredient yang dikenal
                if phrase in self.INGREDIENTS:
                    avoided.add(phrase)
                else:
                    # Cek n-gram dalam phrase
                    for ing in self._sorted_ingredients:
                        if ing in phrase:
                            avoided.add(ing)
                            break
        
        return list(avoided)

    def _extract_main_ingredients(self, text: str, avoid: List[str]) -> List[str]:
        """Ekstrak bahan utama menggunakan n-gram matching."""
        words = text.split()
        found: List[str] = []
        used_positions = set()

        for ing in self._sorted_ingredients:
            ing_words = ing.split()
            n = len(ing_words)
            for i in range(len(words) - n + 1):
                if any(j in used_positions for j in range(i, i + n)):
                    continue
                if words[i:i + n] == ing_words:
                    # Cek synonym
                    canonical = self.INGREDIENT_SYNONYMS.get(ing, ing)
                    if canonical not in avoid:
                        found.append(canonical)
                    for j in range(i, i + n):
                        used_positions.add(j)

        return list(dict.fromkeys(found))

    def _extract_cooking_methods(self, text: str) -> List[str]:
        """Ekstrak teknik memasak."""
        found = []
        for method in self.COOKING_METHODS:
            if method in text:
                found.append(method)
        return list(set(found))

    def _extract_health_conditions(self, text: str) -> List[str]:
        """Ekstrak kondisi kesehatan."""
        found = []
        for keyword in self._sorted_health_keys:
            if keyword in text:
                std_name = self.HEALTH_CONDITIONS[keyword]
                if std_name not in found:
                    found.append(std_name)
        return found

    def _extract_taste_preferences(self, text: str) -> List[str]:
        """Ekstrak preferensi rasa."""
        found = []
        for keyword in self._sorted_taste_keys:
            if keyword in text:
                taste = self.TASTE_KEYWORDS[keyword]
                if taste not in found:
                    found.append(taste)
        return found

    def _extract_time_constraint(self, text: str) -> Optional[int]:
        """Ekstrak waktu memasak dalam menit."""
        # Pola eksplisit: "30 menit", "1 jam"
        m = re.search(r"(\d+)\s*menit", text)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)\s*jam", text)
        if m:
            return int(m.group(1)) * 60

        # Kata kunci waktu relatif
        if any(w in text for w in ["cepat", "cepet", "kilat", "ekspres"]):
            return 30
        if any(w in text for w in ["simpel", "simple", "gampang", "mudah"]):
            return 45
        return None

    def _extract_region(self, text: str) -> Optional[str]:
        """Ekstrak region masakan."""
        for keyword in self._sorted_region_keys:
            if keyword in text:
                return self.REGION_KEYWORDS[keyword]
        return None


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    
    print("=" * 70)
    print("  NER EXTRACTOR TEST")
    print("=" * 70)
    
    # Initialize NER extractor
    ner = NERExtractor()
    
    test_cases = [
        "mau masak ayam goreng yang crispy tapi tanpa tepung",
        "pengen bikin pasta carbonara tapi dairy free karena alergi susu",
        "aku diabetes jadi ga boleh makan yang manis manis",
        "mau yang pedas gurih, direbus aja biar sehat",
        "cariin resep ikan bakar yang cepat dan gampang",
        "kolesterol tinggi jadi ga bisa santan dan minyak goreng",
        "masakan padang yang tidak terlalu pedas",
        "mau masak soto ayam jawa yang enak",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text}")
        print("-" * 50)
        result = ner.extract_all(text)
        print(json.dumps(result, indent=2, ensure_ascii=False))