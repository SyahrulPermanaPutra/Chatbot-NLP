# src/intent_classifier.py
# Intent Classification menggunakan TF-IDF + Random Forest

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import *
from src.preprocessor import TextPreprocessor


class IntentClassifier:
    """
    Classifier untuk mendeteksi intent dari input user
    Menggunakan TF-IDF + Random Forest
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        
    def load_data(self, filepath: str = INTENT_DATASET):
        """Load dataset untuk training"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} training examples")
        print(f"Intent distribution:\n{df['intent'].value_counts()}")
        return df
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Train intent classifier
        Args:
            df: DataFrame with 'text' and 'intent' columns
            test_size: Proportion of test set
        """
        # Preprocess texts
        print("\n[1/5] Preprocessing texts...")
        df['processed_text'] = df['text'].apply(
            lambda x: self.preprocessor.normalize_text(x)
        )
        
        # Split data
        print("[2/5] Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['intent'],
            test_size=test_size,
            random_state=42,
            stratify=df['intent']
        )
        
        # Create TF-IDF vectorizer
        print("[3/5] Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Feature dimensions: {X_train_vec.shape}")
        
        # Train classifier
        print("[4/5] Training Random Forest classifier...")
        self.classifier = RandomForestClassifier(**CLASSIFIER_PARAMS)
        self.classifier.fit(X_train_vec, y_train)
        
        # Evaluate
        print("[5/5] Evaluating model...")
        train_score = self.classifier.score(X_train_vec, y_train)
        test_score = self.classifier.score(X_test_vec, y_test)
        
        print(f"\nTraining accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}")
        
        # Detailed evaluation
        y_pred = self.classifier.predict(X_test_vec)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, text: str, top_k: int = 3):
        """
        Predict intent dari input text
        Args:
            text: User input
            top_k: Return top k predictions with probabilities
        Returns:
            Dictionary with primary intent, confidence, and alternatives
        """
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Model not trained or loaded!")
        
        # Preprocess
        processed = self.preprocessor.normalize_text(text)
        
        # Vectorize
        vec = self.vectorizer.transform([processed])
        
        # Get probabilities
        proba = self.classifier.predict_proba(vec)[0]
        
        # Get top k predictions
        top_indices = np.argsort(proba)[::-1][:top_k]
        top_intents = self.classifier.classes_[top_indices]
        top_probas = proba[top_indices]
        
        # Prepare result
        result = {
            'primary': top_intents[0],
            'confidence': float(top_probas[0]),
            'alternatives': [
                {
                    'intent': intent,
                    'confidence': float(conf)
                }
                for intent, conf in zip(top_intents[1:], top_probas[1:])
                if conf > MIN_CONFIDENCE_THRESHOLD
            ]
        }
        
        return result
    
    def save_model(self, vectorizer_path: str = VECTORIZER_MODEL, 
                   classifier_path: str = INTENT_MODEL):
        """Save trained model"""
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("No model to save!")
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        print(f"Model saved to {vectorizer_path} and {classifier_path}")
    
    def load_model(self, vectorizer_path: str = VECTORIZER_MODEL,
                   classifier_path: str = INTENT_MODEL):
        """Load trained model"""
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(classifier_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        print("Model loaded successfully")


if __name__ == "__main__":
    # Training script
    print("=== Intent Classifier Training ===\n")
    
    classifier = IntentClassifier()
    
    # Load data
    df = classifier.load_data()
    
    # Train
    results = classifier.train(df)
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    classifier.save_model()
    
    # Test predictions
    print("\n=== Testing Predictions ===\n")
    test_inputs = [
        "mau masak ayam goreng yang crispy",
        "aku diabetes jadi ga boleh makan manis",
        "pengen bikin pasta tapi dairy free",
        "thanks ya"
    ]
    
    for text in test_inputs:
        pred = classifier.predict(text)
        print(f"Input: {text}")
        print(f"Intent: {pred['primary']} (confidence: {pred['confidence']:.3f})")
        if pred['alternatives']:
            print(f"Alternatives: {pred['alternatives']}")
        print("-" * 60)
