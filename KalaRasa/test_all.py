#!/usr/bin/env python3
# test_all.py
# Comprehensive testing script untuk semua komponen

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from src.preprocessor import TextPreprocessor
from src.intent_classifier import IntentClassifier
from src.ner_extractor import NERExtractor
from src.nlp_pipeline import RecipeNLPPipeline


def test_preprocessor():
    """Test text preprocessor"""
    print("\n" + "="*80)
    print("TEST 1: TEXT PREPROCESSOR")
    print("="*80)
    
    preprocessor = TextPreprocessor()
    
    test_cases = [
        "gw pengen masak aym gorng yg krispy bgt",
        "mau bikin pasta tp dairy free gmn caranya",
        "aku diabetes jd ga boleh mkn yg manis manis",
        "cariin resep sayur asem donk yg gampang aja"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        result = preprocessor.preprocess(text)
        print(f"  Original:   {result['original']}")
        print(f"  Normalized: {result['normalized']}")
        if result['negations']:
            print(f"  Negations:  {result['negations']}")
    
    print("\n✓ Preprocessor test completed")
    return True


def test_ner_extractor():
    """Test NER extractor"""
    print("\n" + "="*80)
    print("TEST 2: NER EXTRACTOR")
    print("="*80)
    
    ner = NERExtractor()
    
    test_cases = [
        "mau masak ayam goreng yang crispy tanpa tepung",
        "pengen bikin pasta carbonara tapi dairy free",
        "aku diabetes jadi ga boleh makan yang manis",
        "kolesterol tinggi ga bisa santan dan gorengan",
        "mau yang pedas gurih direbus aja"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {text}")
        entities = ner.extract_all(text.lower())
        
        if entities['ingredients']['main']:
            print(f"  Main ingredients: {entities['ingredients']['main']}")
        if entities['ingredients']['avoid']:
            print(f"  Avoid: {entities['ingredients']['avoid']}")
        if entities['cooking_methods']:
            print(f"  Cooking methods: {entities['cooking_methods']}")
        if entities['health_conditions']:
            conditions = [hc['name'] for hc in entities['health_conditions']]
            print(f"  Health conditions: {conditions}")
        if entities['taste_preferences']:
            print(f"  Taste preferences: {entities['taste_preferences']}")
    
    print("\n✓ NER extractor test completed")
    return True


def test_intent_classifier():
    """Test intent classifier"""
    print("\n" + "="*80)
    print("TEST 3: INTENT CLASSIFIER")
    print("="*80)
    
    try:
        classifier = IntentClassifier()
        classifier.load_model()
        
        test_cases = [
            ("mau masak ayam goreng", "cari_resep"),
            ("aku diabetes ga boleh gula", "informasi_kondisi_kesehatan"),
            ("pengen yang pedas banget", "informasi_preferensi"),
            ("santan bisa diganti apa", "tanya_alternatif"),
            ("terima kasih", "chitchat")
        ]
        
        correct = 0
        for i, (text, expected) in enumerate(test_cases, 1):
            pred = classifier.predict(text)
            is_correct = pred['primary'] == expected
            correct += is_correct
            
            status = "✓" if is_correct else "✗"
            print(f"\nTest {i}: {status}")
            print(f"  Input: {text}")
            print(f"  Expected: {expected}")
            print(f"  Predicted: {pred['primary']} (confidence: {pred['confidence']:.3f})")
        
        accuracy = correct / len(test_cases) * 100
        print(f"\n✓ Intent classifier test completed")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")
        return True
        
    except Exception as e:
        print(f"\n✗ Intent classifier test failed: {e}")
        print("  Please run train_model.py first!")
        return False


def test_full_pipeline():
    """Test full NLP pipeline"""
    print("\n" + "="*80)
    print("TEST 4: FULL NLP PIPELINE")
    print("="*80)
    
    try:
        pipeline = RecipeNLPPipeline(load_models=True)
        
        test_cases = [
            "gw pengen masak ayam goreng yang krispy banget tapi gak pake tepung",
            "mau bikin pasta carbonara tapi dairy free gimana caranya",
            "aku diabetes jadi ga boleh makan yang manis manis",
            "cariin resep ikan bakar yang cepat dong",
            "kolesterol tinggi nih ga bisa santan dan gorengan"
        ]
        
        for i, text in enumerate(test_cases, 1):
            print(f"\n{'─'*80}")
            print(f"Test Case {i}")
            print(f"{'─'*80}")
            print(f"Input: {text}\n")
            
            result = pipeline.process(text)
            
            # Show summary
            print("Summary:")
            print(pipeline.get_summary(result))
            
            # Verify output structure
            assert 'intent' in result
            assert 'entities' in result
            assert 'constraints' in result
            assert 'metadata' in result
            
            print("✓ Output structure valid")
        
        print(f"\n✓ Full pipeline test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        return False


def test_json_output():
    """Test JSON output format"""
    print("\n" + "="*80)
    print("TEST 5: JSON OUTPUT FORMAT")
    print("="*80)
    
    try:
        pipeline = RecipeNLPPipeline(load_models=True)
        
        text = "mau masak ayam goreng yang krispy tapi tanpa tepung untuk diet"
        result = pipeline.process(text)
        
        print("\nSample JSON Output:")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:500] + "...")
        
        # Validate JSON structure
        required_keys = [
            'version', 'timestamp', 'user_input', 'normalized_input',
            'intent', 'entities', 'constraints', 'metadata'
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        print("\n✓ JSON output format valid")
        return True
        
    except Exception as e:
        print(f"\n✗ JSON output test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print(" RECIPE NLP CHATBOT - COMPREHENSIVE TESTING")
    print("="*80)
    
    results = {
        'Preprocessor': test_preprocessor(),
        'NER Extractor': test_ner_extractor(),
        'Intent Classifier': test_intent_classifier(),
        'Full Pipeline': test_full_pipeline(),
        'JSON Output': test_json_output()
    }
    
    # Summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} - {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        if not results['Intent Classifier']:
            print("   → Run train_model.py first to train the intent classifier")


if __name__ == "__main__":
    main()
