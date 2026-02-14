# train_model.py
# Script untuk training intent classifier

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.intent_classifier import IntentClassifier
from config.config import MODEL_DIR, INTENT_DATASET


def main():
    """Main training function"""
    print("="*80)
    print(" RECIPE NLP CHATBOT - MODEL TRAINING")
    print("="*80)
    print()
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize classifier
    print("[1/4] Initializing intent classifier...")
    classifier = IntentClassifier()
    
    # Load dataset
    print(f"[2/4] Loading dataset from {INTENT_DATASET}...")
    df = classifier.load_data(INTENT_DATASET)
    print(f"      Loaded {len(df)} training examples")
    print(f"      Unique intents: {df['intent'].nunique()}")
    print()
    
    # Train model
    print("[3/4] Training model (this may take a while)...")
    results = classifier.train(df, test_size=0.2)
    print()
    
    # Save model
    print("[4/4] Saving trained model...")
    classifier.save_model()
    print()
    
    # Summary
    print("="*80)
    print(" TRAINING COMPLETED")
    print("="*80)
    print(f"Train Accuracy: {results['train_score']:.4f}")
    print(f"Test Accuracy:  {results['test_score']:.4f}")
    print()
    print("Model files saved:")
    print(f"  - {MODEL_DIR}/intent_classifier.pkl")
    print(f"  - {MODEL_DIR}/tfidf_vectorizer.pkl")
    print()
    
if __name__ == "__main__":
    main()
