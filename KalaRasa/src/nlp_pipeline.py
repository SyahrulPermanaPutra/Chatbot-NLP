# src/nlp_pipeline.py
# Main NLP Pipeline yang mengintegrasikan preprocessing, intent, dan NER

import json
import time
from datetime import datetime
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import OUTPUT_FORMAT
from src.preprocessor import TextPreprocessor
from src.intent_classifier import IntentClassifier
from src.ner_extractor import NERExtractor


class RecipeNLPPipeline:
    """
    Main NLP Pipeline untuk Recipe Chatbot
    Menggabungkan: Preprocessing -> Intent Classification -> NER
    """
    
    def __init__(self, load_models: bool = True):
        """
        Initialize pipeline
        Args:
            load_models: If True, load pre-trained models
        """
        print("Initializing Recipe NLP Pipeline...")
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.intent_classifier = IntentClassifier()
        self.ner_extractor = NERExtractor()
        
        # Load models if specified
        if load_models:
            try:
                self.intent_classifier.load_model()
                print("✓ Intent classifier loaded")
            except Exception as e:
                print(f"✗ Could not load intent classifier: {e}")
                print("  Run training first!")
        
        print("Pipeline ready!\n")
    
    def process(self, user_input: str) -> Dict:
        """
        Process user input through full pipeline
        Args:
            user_input: Raw text dari user
        Returns:
            Structured JSON output
        """
        start_time = time.time()
        
        # 1. Preprocessing
        preprocessed = self.preprocessor.preprocess(user_input)
        normalized_text = preprocessed['normalized']
        
        # 2. Intent Classification
        try:
            intent_result = self.intent_classifier.predict(normalized_text)
        except Exception as e:
            print(f"Intent classification error: {e}")
            intent_result = {
                'primary': 'unknown',
                'confidence': 0.0,
                'alternatives': []
            }
        
        # 3. NER - Entity Extraction
        entities = self.ner_extractor.extract_all(normalized_text)
        
        # 4. Build structured output
        output = self._build_output(
            user_input=user_input,
            normalized_input=normalized_text,
            intent=intent_result,
            entities=entities,
            processing_time=time.time() - start_time
        )
        
        return output
    
    def _build_output(self, user_input: str, normalized_input: str,
                     intent: Dict, entities: Dict, processing_time: float) -> Dict:
        """Build structured JSON output"""
        
        output = OUTPUT_FORMAT.copy()
        output['timestamp'] = datetime.now().isoformat()
        output['user_input'] = user_input
        output['normalized_input'] = normalized_input
        
        # Intent
        output['intent']['primary'] = intent['primary']
        output['intent']['confidence'] = intent['confidence']
        output['intent']['secondary'] = intent.get('alternatives', [])
        
        # Entities
        output['entities']['ingredients'] = entities['ingredients']
        output['entities']['cooking_methods'] = entities.get('cooking_methods', [])
        output['entities']['health_conditions'] = [
            hc['name'] for hc in entities.get('health_conditions', [])
        ]
        output['entities']['taste_preferences'] = entities.get('taste_preferences', [])
        output['entities']['time_constraint'] = entities.get('time_constraint')
        
        # Constraints (compiled dari berbagai sumber)
        output['constraints']['must_include'] = entities['ingredients']['main']
        output['constraints']['must_exclude'] = entities['ingredients']['avoid']
        
        # Add dietary restrictions dari health conditions
        for hc in entities.get('health_conditions', []):
            output['constraints']['dietary_restrictions'].append({
                'condition': hc['name'],
                'avoid': hc['avoid'],
                'recommended': hc.get('recommended', [])
            })
        
        # Metadata
        output['metadata']['processing_time'] = round(processing_time, 4)
        output['metadata']['confidence_scores'] = {
            'intent': intent['confidence']
        }
        
        return output
    
    def process_batch(self, inputs: list) -> list:
        """Process multiple inputs"""
        results = []
        for inp in inputs:
            result = self.process(inp)
            results.append(result)
        return results
    
    def get_summary(self, output: Dict) -> str:
        """
        Generate human-readable summary dari output
        Useful untuk debugging
        """
        intent = output['intent']['primary']
        confidence = output['intent']['confidence']
        
        summary = f"Intent: {intent} ({confidence:.2%})\n"
        
        # Ingredients
        if output['entities']['ingredients']['main']:
            summary += f"Main ingredients: {', '.join(output['entities']['ingredients']['main'])}\n"
        
        if output['entities']['ingredients']['avoid']:
            summary += f"Avoid: {', '.join(output['entities']['ingredients']['avoid'])}\n"
        
        # Cooking methods
        if output['entities']['cooking_methods']:
            summary += f"Cooking methods: {', '.join(output['entities']['cooking_methods'])}\n"
        
        # Health conditions
        if output['entities']['health_conditions']:
            summary += f"Health conditions: {', '.join(output['entities']['health_conditions'])}\n"
        
        # Taste
        if output['entities']['taste_preferences']:
            summary += f"Taste preferences: {', '.join(output['entities']['taste_preferences'])}\n"
        
        return summary


if __name__ == "__main__":
    # Testing pipeline
    print("=== Recipe NLP Pipeline Test ===\n")
    
    # Initialize pipeline
    pipeline = RecipeNLPPipeline(load_models=True)
    
    # Test cases
    test_inputs = [
        "gw pengen masak ayam goreng yang krispy banget tapi gak pake tepung",
        "mau bikin pasta carbonara tapi dairy free gimana caranya",
        "aku diabetes jadi ga boleh makan yang manis manis",
        "cariin resep ikan bakar yang cepat dan gampang dong",
        "kolesterol tinggi nih, ga bisa santan dan gorengan",
        "pengen yang pedas gurih, direbus aja biar sehat"
    ]
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\n{'='*80}")
        print(f"Test Case #{i}")
        print(f"{'='*80}")
        print(f"Input: {text}\n")
        
        # Process
        result = pipeline.process(text)
        
        # Show summary
        print("Summary:")
        print(pipeline.get_summary(result))
        
        # Show full JSON (pretty print)
        print("\nFull Output:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
