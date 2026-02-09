# src/preprocessor.py
# Text preprocessing untuk normalisasi input user

import re
import json
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KB_NORMALIZATION


class TextPreprocessor:
    """
    Kelas untuk preprocessing teks input user
    - Normalisasi kata informal
    - Koreksi typo
    - Cleaning
    """
    
    def __init__(self):
        self.normalization_dict = self._load_normalization_dict()
        
    def _load_normalization_dict(self) -> Dict:
        """Load knowledge base normalisasi"""
        try:
            with open(KB_NORMALIZATION, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load normalization dict: {e}")
            return {"normalisasi_informal": {}, "typo_umum": {}}
    
    def normalize_text(self, text: str) -> str:
        """
        Normalisasi teks lengkap
        Args:
            text: Input text dari user
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters tapi pertahankan huruf dan angka
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalisasi kata informal
        text = self._normalize_informal_words(text)
        
        # Fix typo umum
        text = self._fix_common_typos(text)
        
        # Remove extra whitespace lagi
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _normalize_informal_words(self, text: str) -> str:
        """Normalisasi kata-kata informal ke formal"""
        words = text.split()
        normalized_words = []
        
        informal_dict = self.normalization_dict.get('normalisasi_informal', {})
        
        for word in words:
            # Cek exact match
            if word in informal_dict:
                normalized_words.append(informal_dict[word])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def _fix_common_typos(self, text: str) -> str:
        """Fix typo umum pada nama bahan makanan"""
        words = text.split()
        fixed_words = []
        
        typo_dict = self.normalization_dict.get('typo_umum', {})
        
        for word in words:
            if word in typo_dict:
                fixed_words.append(typo_dict[word])
            else:
                fixed_words.append(word)
        
        return ' '.join(fixed_words)
    
    def extract_negations(self, text: str) -> List[str]:
        """
        Ekstrak kata-kata yang menunjukkan penolakan/pantangan
        Returns: List of negation patterns found
        """
        negation_patterns = [
            r'tidak\s+\w+',
            r'tanpa\s+\w+',
            r'gak\s+\w+',
            r'ga\s+\w+',
            r'jangan\s+\w+',
            r'hindari\s+\w+',
            r'ga\s+boleh\s+\w+',
            r'gak\s+bisa\s+\w+'
        ]
        
        found_negations = []
        for pattern in negation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_negations.extend(matches)
        
        return found_negations
    
    def preprocess(self, text: str) -> Dict:
        """
        Main preprocessing function
        Returns dictionary with original and normalized text
        """
        return {
            'original': text,
            'normalized': self.normalize_text(text),
            'negations': self.extract_negations(text)
        }


if __name__ == "__main__":
    # Testing
    preprocessor = TextPreprocessor()
    
    test_cases = [
        "gw pengen masak aym gorng yg krispy bgt tapi gak pake tepung",
        "mau bikin pasta carbonara tp dairy free gimana caranya",
        "aku diabetes jd ga boleh makan yg manis manis",
        "cariin resep sayur asem donk yg gampang"
    ]
    
    print("=== Text Preprocessing Test ===\n")
    for test in test_cases:
        result = preprocessor.preprocess(test)
        print(f"Original: {result['original']}")
        print(f"Normalized: {result['normalized']}")
        print(f"Negations: {result['negations']}")
        print("-" * 60)
