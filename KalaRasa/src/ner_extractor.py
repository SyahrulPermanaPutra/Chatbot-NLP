# src/ner_extractor.py
# Named Entity Recognition untuk ekstraksi entities dari input

import json
import re
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import *


class NERExtractor:
    """
    Rule-based + Pattern matching NER untuk ekstraksi entities
    Entity types:
    - BAHAN_UTAMA: Bahan makanan utama
    - BAHAN_TAMBAHAN: Bahan pendukung
    - BAHAN_HINDARI: Bahan yang harus dihindari
    - TEKNIK_MASAK: Cara memasak
    - KONDISI_KESEHATAN: Kondisi kesehatan pengguna
    - PREFERENSI_RASA: Preferensi rasa
    """
    
    def __init__(self):
        self.kb_ingredients = self._load_json(KB_INGREDIENTS)
        self.kb_cooking = self._load_json(KB_COOKING_METHODS)
        self.kb_health = self._load_json(KB_HEALTH_CONDITIONS)
        
        # Build lookup dictionaries
        self.ingredient_lookup = self._build_ingredient_lookup()
        self.cooking_lookup = self._build_cooking_lookup()
        self.health_lookup = self._build_health_lookup()
        self.taste_lookup = self._build_taste_lookup()
        
    def _load_json(self, filepath: str) -> Dict:
        """Load JSON knowledge base"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}
    
    def _build_ingredient_lookup(self) -> Dict[str, str]:
        """Build flat dictionary untuk ingredient lookup"""
        lookup = {}
        for category, subcategories in self.kb_ingredients.items():
            for subcat, items in subcategories.items():
                for item in items:
                    lookup[item.lower()] = category
        return lookup
    
    def _build_cooking_lookup(self) -> Dict[str, str]:
        """Build flat dictionary untuk cooking method lookup"""
        lookup = {}
        for category, subcategories in self.kb_cooking.items():
            for subcat, methods in subcategories.items():
                for method in methods:
                    lookup[method.lower()] = subcat
        return lookup
    
    def _build_health_lookup(self) -> Dict[str, str]:
        """Build dictionary untuk health condition lookup"""
        lookup = {}
        conditions = self.kb_health.get('kondisi_kesehatan', {})
        for condition, data in conditions.items():
            # Add main name
            lookup[data['nama'].lower()] = data['nama']
            # Add synonyms
            for synonym in data.get('sinonim', []):
                lookup[synonym.lower()] = data['nama']
        return lookup
    
    def _build_taste_lookup(self) -> Dict[str, str]:
        """Build dictionary untuk taste preference lookup"""
        lookup = {}
        prefs = self.kb_health.get('preferensi_rasa', {})
        for taste, keywords in prefs.items():
            for keyword in keywords:
                lookup[keyword.lower()] = taste
        return lookup
    
    def extract_ingredients(self, text: str) -> Dict[str, List[str]]:
        """
        Extract ingredients dari text
        Returns: Dict dengan main, additional, dan avoid ingredients
        """
        words = text.split()
        main_ingredients = []
        avoid_ingredients = []
        
        # Pattern untuk mendeteksi ingredients yang harus dihindari
        avoid_patterns = [
            r'tanpa\s+(\w+)',
            r'tidak\s+pakai\s+(\w+)',
            r'tidak\s+(\w+)',
            r'ga\s+pakai\s+(\w+)',
            r'ga\s+boleh\s+(\w+)',
            r'hindari\s+(\w+)',
            r'gak\s+bisa\s+(\w+)'
        ]
        
        # Extract ingredients to avoid
        for pattern in avoid_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match in self.ingredient_lookup:
                    avoid_ingredients.append(match)
        
        # Extract main ingredients (n-gram approach)
        for n in range(3, 0, -1):  # Check 3-gram, 2-gram, 1-gram
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                if ngram in self.ingredient_lookup and ngram not in avoid_ingredients:
                    main_ingredients.append(ngram)
        
        return {
            'main': list(set(main_ingredients)),
            'additional': [],  # Could be expanded with more complex logic
            'avoid': list(set(avoid_ingredients))
        }
    
    def extract_cooking_methods(self, text: str) -> List[str]:
        """Extract cooking methods dari text"""
        words = text.split()
        methods = []
        
        # N-gram approach
        for n in range(3, 0, -1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                if ngram in self.cooking_lookup:
                    methods.append(ngram)
        
        return list(set(methods))
    
    def extract_health_conditions(self, text: str) -> List[Dict]:
        """
        Extract health conditions dan pantangannya
        Returns: List of dicts with condition name and restrictions
        """
        conditions = []
        
        # Check for health condition mentions
        for term, condition_name in self.health_lookup.items():
            if term in text:
                # Get full condition data
                cond_data = None
                for cond_key, data in self.kb_health.get('kondisi_kesehatan', {}).items():
                    if data['nama'] == condition_name:
                        cond_data = data
                        break
                
                if cond_data:
                    conditions.append({
                        'name': condition_name,
                        'avoid': cond_data.get('hindari', []),
                        'recommended': cond_data.get('anjuran', [])
                    })
        
        return conditions
    
    def extract_taste_preferences(self, text: str) -> List[str]:
        """Extract taste preferences"""
        preferences = []
        
        for term, taste in self.taste_lookup.items():
            if term in text:
                preferences.append(taste)
        
        # Handle negations
        negation_patterns = [
            r'tidak\s+pedas',
            r'ga\s+pedas',
            r'gak\s+pedas'
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, text):
                if 'pedas' in preferences:
                    preferences.remove('pedas')
        
        return list(set(preferences))
    
    def extract_time_constraint(self, text: str) -> str:
        """Extract time constraints if mentioned"""
        time_patterns = [
            (r'(\d+)\s*menit', 'minutes'),
            (r'cepat', 'quick'),
            (r'cepet', 'quick'),
            (r'simple', 'simple'),
            (r'simpel', 'simple'),
            (r'gampang', 'easy'),
            (r'mudah', 'easy')
        ]
        
        for pattern, category in time_patterns:
            match = re.search(pattern, text)
            if match:
                if category == 'minutes':
                    return f"{match.group(1)} minutes"
                else:
                    return category
        
        return None
    
    def extract_all(self, text: str) -> Dict:
        """
        Extract all entities dari text
        Returns: Comprehensive entity dictionary
        """
        text = text.lower()
        
        ingredients = self.extract_ingredients(text)
        cooking_methods = self.extract_cooking_methods(text)
        health_conditions = self.extract_health_conditions(text)
        taste_prefs = self.extract_taste_preferences(text)
        time_constraint = self.extract_time_constraint(text)
        
        # Compile avoid list dari health conditions
        all_avoid = ingredients['avoid'].copy()
        for condition in health_conditions:
            all_avoid.extend(condition['avoid'])
        
        return {
            'ingredients': {
                'main': ingredients['main'],
                'additional': ingredients['additional'],
                'avoid': list(set(all_avoid))
            },
            'cooking_methods': cooking_methods,
            'health_conditions': health_conditions,
            'taste_preferences': taste_prefs,
            'time_constraint': time_constraint
        }


if __name__ == "__main__":
    # Testing
    print("=== NER Extractor Test ===\n")
    
    ner = NERExtractor()
    
    test_cases = [
        "mau masak ayam goreng yang crispy tapi tanpa tepung",
        "pengen bikin pasta carbonara tapi dairy free karena alergi susu",
        "aku diabetes jadi ga boleh makan yang manis manis",
        "mau yang pedas gurih, direbus aja biar sehat",
        "cariin resep ikan bakar yang cepat dan gampang",
        "kolesterol tinggi jadi ga bisa santan dan gorengan"
    ]
    
    for text in test_cases:
        print(f"Input: {text}")
        entities = ner.extract_all(text)
        print(json.dumps(entities, indent=2, ensure_ascii=False))
        print("-" * 80)
