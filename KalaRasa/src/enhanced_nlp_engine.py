# src/nlp_pipeline.py - FIXED VERSION
# Enhanced NLP Engine dengan safety-first approach

import re
import json
from typing import Dict, List, Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import TextPreprocessor
from src.intent_classifier import IntentClassifier
from src.ner_extractor import NERExtractor


class EnhancedNLPEngine:
    """
    Enhanced NLP Engine dengan prinsip:
    1. Klarifikasi > Error
    2. Keamanan > Kelengkapan
    3. Jangan berasumsi
    """
    
    # Confidence thresholds
    MIN_CONFIDENCE = 0.35  
    
    # Status codes
    STATUS_OK = "ok"
    STATUS_FALLBACK = "fallback"
    STATUS_CLARIFICATION = "clarification"
    
    # Actions
    ACTION_MATCH_RECIPE = "match_recipe"
    ACTION_ASK_CLARIFICATION = "ask_clarification"
    ACTION_REJECT_INPUT = "reject_input"
    
    def __init__(self):
        """Initialize engine with error handling"""
        print("Initializing NLP components...")
        
        try:
            self.preprocessor = TextPreprocessor()
            print("  ✓ Preprocessor loaded")
        except Exception as e:
            print(f"  ✗ Preprocessor failed: {e}")
            raise
        
        try:
            self.intent_classifier = IntentClassifier()
            self.intent_classifier.load_model()
            print("  ✓ Intent classifier loaded")
        except FileNotFoundError:
            print("  ✗ Model files not found!")
            print("  Please run: python train_model.py")
            raise
        except Exception as e:
            print(f"  ✗ Intent classifier failed: {e}")
            raise
        
        try:
            self.ner_extractor = NERExtractor()
            print("  ✓ NER extractor loaded")
        except Exception as e:
            print(f"  ✗ NER extractor failed: {e}")
            raise
    
    def process(self, user_input: str) -> Dict:
        """
        Main processing function dengan improvements
        """
        try:
            # Step 0: Handle empty/single character
            if not user_input or len(user_input.strip()) == 0:
                return self._create_response(
                    status=self.STATUS_FALLBACK,
                    intent="unknown",
                    confidence=0.0,
                    entities={},
                    action=self.ACTION_ASK_CLARIFICATION,
                    message="Silakan tanyakan sesuatu tentang resep masakan 😊"
                )
            
            # Step 0.5: Coba handle input sederhana dulu
            simple_response = self._handle_simple_input(user_input, user_input.lower())
            if simple_response:
                return simple_response
            
            # Step 1: Gibberish detection (yang sudah diperbaiki)
            if self._is_gibberish(user_input):
                # Tapi coba dulu dengan preprocessing
                preprocessed = self.preprocessor.preprocess(user_input)
                normalized_text = preprocessed['normalized']
            
                # Coba klasifikasi meski dianggap gibberish
                intent_result = self.intent_classifier.predict(normalized_text)
            
                if intent_result['confidence'] > 0.3:  # Threshold rendah
                    # Lanjutkan processing
                    intent = intent_result['primary']
                    confidence = intent_result['confidence']
                
                    # Skip ke entity extraction
                    entities = self.ner_extractor.extract_all(normalized_text)
                
                    return self._create_response(
                        status=self.STATUS_OK,
                        intent=intent,
                        confidence=confidence,
                        entities=entities,
                        action=self.ACTION_MATCH_RECIPE,
                        message="Mencari resep sesuai permintaan..."
                    )
                else:
                    return self._create_response(
                        status=self.STATUS_FALLBACK,
                        intent="unknown",
                        confidence=0.0,
                        entities={},
                        action=self.ACTION_REJECT_INPUT,
                        message="Aku belum bisa memahami maksud kamu 😅 Coba tulis dengan kalimat sederhana, misalnya: 'mau masak ayam' atau 'resep ikan'"
                    )

            # Step 2: Preprocessing
            preprocessed = self.preprocessor.preprocess(user_input)
            normalized_text = preprocessed['normalized']
            
            # Step 3: Intent classification
            intent_result = self.intent_classifier.predict(normalized_text)
            intent = intent_result['primary']
            confidence = intent_result['confidence']
            
            # Step 4: Check confidence threshold
            if confidence < self.MIN_CONFIDENCE:
                return self._create_response(
                    status=self.STATUS_FALLBACK,
                    intent="unknown",
                    confidence=confidence,
                    entities={},
                    action=self.ACTION_ASK_CLARIFICATION,
                    message="Maaf, aku belum yakin maksud kamu. Kamu bisa coba:\n• Cari resep (contoh: mau masak ayam goreng)\n• Tanya bahan masakan\n• Tanya pantangan makanan"
                )
            
            # Step 5: Entity extraction
            entities = self.ner_extractor.extract_all(normalized_text)
            
            # Step 6: Slot validation untuk intent cari_resep
            if intent in ['cari_resep', 'cari_resep_kompleks', 'cari_resep_kondisi']:
                validation_result = self._validate_recipe_search_slots(entities)
                if validation_result['needs_clarification']:
                    return self._create_response(
                        status=self.STATUS_CLARIFICATION,
                        intent=intent,
                        confidence=confidence,
                        entities=entities,
                        action=self.ACTION_ASK_CLARIFICATION,
                        message=validation_result['message']
                    )
            
            # Step 7: Safety validation untuk health conditions
            if entities.get('health_conditions'):
                safety_result = self._validate_health_safety(entities)
                if not safety_result['is_safe']:
                    return self._create_response(
                        status=self.STATUS_FALLBACK,
                        intent=intent,
                        confidence=confidence,
                        entities=entities,
                        action=self.ACTION_ASK_CLARIFICATION,
                        message=safety_result['message']
                    )
            
            # Step 8: All checks passed - ready for recipe matching
            return self._create_response(
                status=self.STATUS_OK,
                intent=intent,
                confidence=confidence,
                entities=entities,
                action=self.ACTION_MATCH_RECIPE,
                message="Processing recipe search..."
            )
        
        except Exception as e:
            # Catch-all error handler
            print(f"Error in process(): {e}")
            import traceback
            traceback.print_exc()
            
            return self._create_response(
                status=self.STATUS_FALLBACK,
                intent="error",
                confidence=0.0,
                entities={},
                action=self.ACTION_REJECT_INPUT,
                message=f"Terjadi kesalahan sistem. Silakan coba lagi."
            )
    
    def _is_gibberish(self, text: str) -> bool:
        try:
           text = text.strip()
        
            # Rule 0: Angka saja tidak dianggap gibberish (bisa jadi "1", "2", dll)
           if text.isdigit():
            return False
            
            # Rule 1: Check minimum length (jangan terlalu ketat)
            if len(text) < 2:  # 1 karakter pasti gibberish
                return True
            
            # Rule 2: Check alphabet ratio (lebih longgar)
            alpha_chars = sum(c.isalpha() for c in text)
            total_chars = len(text.replace(' ', ''))
        
            if total_chars == 0:
                return True
            
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio < 0.3:  # Turun dari 0.5 menjadi 0.3
                return True
            
            # Rule 3: Check meaningful words (lebih permisif)
            words = text.lower().split()
        
            # Filter out very short words (toleransi lebih tinggi)
            meaningful_words = [w for w in words if len(w) >= 2]
        
            # Untuk input sangat pendek, lebih toleran
            if len(words) == 1 and len(words[0]) >= 3:
                return False
            
            if len(meaningful_words) < 1:  # Turun dari 2 menjadi 1
                return True
            
            # Rule 4: Check vocabulary recognition (lebih rendah threshold)
            recognized = self._check_vocabulary_recognition(meaningful_words)
            if recognized < 0.2:  # Turun dari 0.3 menjadi 0.2
                return True
            
            return False
        
        except Exception as e:
            print(f"Error in _is_gibberish: {e}")
            return False  # Jika error, berikan benefit of doubt

    # Tambahkan method untuk handle input pendek/sederhana
    def _handle_simple_input(self, text: str, normalized_text: str) -> Optional[Dict]:
        """
        Handle input yang sangat pendek/sederhana dengan logika khusus
        """
        simple_keywords = {
            # Kata kunci -> intent
            'ayam': 'cari_resep',
            'ikan': 'cari_resep', 
            'sapi': 'cari_resep',
            'tempe': 'cari_resep',
            'tahu': 'cari_resep',
            'nasi': 'cari_resep',
            'mie': 'cari_resep',
            'pasta': 'cari_resep',
            'makan': 'cari_resep',
            'masak': 'cari_resep',
            'resep': 'cari_resep',
            'bikin': 'cari_resep',
            'buat': 'cari_resep',
            'mau': 'cari_resep',
            'pengen': 'cari_resep',
            'ingin': 'cari_resep',
        }
    
        words = normalized_text.split()
    
        # Cek jika ada kata kunci sederhana
        for word in words:
            if word in simple_keywords:
                return self._create_response(
                    status=self.STATUS_OK,
                    intent=simple_keywords[word],
                    confidence=0.65,  # Confidence cukup
                    entities={'ingredients': {'main': [word] if word in ['ayam', 'ikan', 'sapi', 'tempe', 'tahu'] else []}},
                    action=self.ACTION_MATCH_RECIPE,
                    message="Mencari resep..."
                )
    
        return None

    
    def _check_vocabulary_recognition(self, words: List[str]) -> float:
        """
        Check what percentage of words are recognized
        Simple check against common Indonesian words
        """
        # Common Indonesian words + cooking terms
        common_words = {
            # Pronouns & common words
            'saya', 'aku', 'gw', 'gue', 'kamu', 'mau', 'ingin', 'pengen',
            'ada', 'tidak', 'ga', 'gak', 'bisa', 'boleh', 'harus',
            
            # Cooking verbs
            'masak', 'bikin', 'buat', 'goreng', 'rebus', 'panggang', 'tumis',
            
            # Ingredients
            'ayam', 'ikan', 'sapi', 'udang', 'sayur', 'tempe', 'tahu',
            'nasi', 'mie', 'pasta', 'telur', 'daging',
            
            # Cooking terms
            'resep', 'masakan', 'bahan', 'bumbu', 'cepat', 'mudah', 'simple',
            
            # Health terms
            'diabetes', 'kolesterol', 'diet', 'sehat', 'alergi',
            
            # Taste
            'pedas', 'manis', 'asin', 'gurih', 'segar',
            
            # Common words
            'yang', 'dengan', 'untuk', 'tanpa', 'aja', 'dong', 'sih'
        }
        
        if len(words) == 0:
            return 0.0
        
        recognized_count = sum(1 for word in words if word in common_words)
        return recognized_count / len(words)
    
    def _validate_recipe_search_slots(self, entities: Dict) -> Dict:
        """
        Validate if enough information for recipe search
        
        Returns:
            {
                'needs_clarification': bool,
                'message': str
            }
        """
        try:
            ingredients = entities.get('ingredients', {})
            main_ingredients = ingredients.get('main', [])
            
            # Check if ingredients are specified
            if not main_ingredients:
                return {
                    'needs_clarification': True,
                    'message': "Kamu mau masak bahan apa?\n• Ayam\n• Ikan\n• Sayur\n• Daging\n• Seafood\n• Lainnya"
                }
            
            return {
                'needs_clarification': False,
                'message': ""
            }
        except Exception as e:
            print(f"Error in _validate_recipe_search_slots: {e}")
            return {
                'needs_clarification': False,
                'message': ""
            }
    
    def _validate_health_safety(self, entities: Dict) -> Dict:
        """
        Validate health safety requirements
        
        Returns:
            {
                'is_safe': bool,
                'message': str
            }
        """
        try:
            health_conditions = entities.get('health_conditions', [])
            
            if not health_conditions:
                return {'is_safe': True, 'message': ''}
            
            # Extract all restrictions
            all_restrictions = []
            for condition in health_conditions:
                if isinstance(condition, dict):
                    all_restrictions.extend(condition.get('avoid', []))
            
            # If too many restrictions (potential issue)
            if len(all_restrictions) > 20:
                return {
                    'is_safe': False,
                    'message': "Aku mendeteksi banyak pantangan makanan. Untuk keamanan, sebaiknya konsultasi dengan ahli gizi untuk rekomendasi yang tepat."
                }
            
            return {'is_safe': True, 'message': ''}
        except Exception as e:
            print(f"Error in _validate_health_safety: {e}")
            return {'is_safe': True, 'message': ''}
    
    def _create_response(
        self,
        status: str,
        intent: str,
        confidence: float,
        entities: Dict,
        action: str,
        message: str
    ) -> Dict:
        """
        Create structured JSON response
        
        Output format:
        {
            "status": "ok | fallback | clarification",
            "intent": "...",
            "confidence": 0.xx,
            "entities": {...},
            "action": "match_recipe | ask_clarification | reject_input",
            "message": "pesan ke user"
        }
        """
        return {
            "status": status,
            "intent": intent,
            "confidence": round(confidence, 2),
            "entities": self._clean_entities(entities),
            "action": action,
            "message": message
        }
    
    def _clean_entities(self, entities: Dict) -> Dict:
        """Clean entities - remove empty fields"""
        try:
            cleaned = {}
            
            if 'ingredients' in entities:
                ing = entities['ingredients']
                cleaned['ingredients'] = {
                    'main': ing.get('main', []),
                    'avoid': ing.get('avoid', [])
                }
            
            if 'cooking_methods' in entities and entities['cooking_methods']:
                cleaned['cooking_methods'] = entities['cooking_methods']
            
            if 'taste_preferences' in entities and entities['taste_preferences']:
                cleaned['taste_preferences'] = entities['taste_preferences']
            
            if 'health_conditions' in entities and entities['health_conditions']:
                cleaned['health_conditions'] = [
                    hc['name'] if isinstance(hc, dict) else hc 
                    for hc in entities['health_conditions']
                ]
            
            if 'time_constraint' in entities and entities['time_constraint']:
                cleaned['time_constraint'] = entities['time_constraint']
            
            return cleaned
        except Exception as e:
            print(f"Error in _clean_entities: {e}")
            return {}


if __name__ == "__main__":
    # Testing
    print("=== Enhanced NLP Engine Test ===\n")
    
    try:
        engine = EnhancedNLPEngine()
        
        test_cases = [
            # Valid inputs
            ("mau masak ayam goreng", "Should work"),
            ("aku diabetes ga boleh gula", "Should work with health check"),
            ("pengen bikin pasta", "Should work"),
            
            # Missing ingredients
            ("mau masak yang enak", "Should ask for ingredient"),
            
            # Low confidence
            ("xyz abc def", "Should reject as gibberish"),
            ("12345 !@#$%", "Should reject as gibberish"),
            
            # Too vague
            ("halo", "Should ask for clarification"),
            ("iya", "Should ask for clarification"),
        ]
        
        for query, expected in test_cases:
            print(f"\n{'='*80}")
            print(f"Input: {query}")
            print(f"Expected: {expected}")
            print(f"{'='*80}")
            
            result = engine.process(query)
            
            print(f"\nStatus: {result['status']}")
            print(f"Intent: {result['intent']} (confidence: {result['confidence']})")
            print(f"Action: {result['action']}")
            print(f"Message: {result['message']}")
            
            if result['entities']:
                print(f"Entities: {json.dumps(result['entities'], indent=2, ensure_ascii=False)}")
    
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()