# config.py
# Configuration file untuk Recipe NLP Chatbot

import os

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Knowledge Base paths
KB_INGREDIENTS = os.path.join(DATA_DIR, 'knowledge_base_ingredients.json')
KB_COOKING_METHODS = os.path.join(DATA_DIR, 'knowledge_base_cooking_methods.json')
KB_HEALTH_CONDITIONS = os.path.join(DATA_DIR, 'knowledge_base_health_conditions.json')
KB_NORMALIZATION = os.path.join(DATA_DIR, 'knowledge_base_normalization.json')

# Dataset paths
INTENT_DATASET = os.path.join(DATA_DIR, 'intent_dataset.csv')
NER_DATASET = os.path.join(DATA_DIR, 'ner_dataset.csv')
RECIPE_DATABASE = os.path.join(DATA_DIR, 'recipe_database.json')

# Model paths
INTENT_MODEL = os.path.join(MODEL_DIR, 'intent_classifier.pkl')
VECTORIZER_MODEL = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
NER_MODEL = os.path.join(MODEL_DIR, 'ner_model.pkl')

# Intent labels
INTENT_LABELS = [
    'cari_resep',
    'cari_resep_kondisi',
    'cari_resep_pantangan',
    'cari_resep_kompleks',
    'informasi_kondisi_kesehatan',
    'informasi_pantangan',
    'informasi_preferensi',
    'tanya_alternatif',
    'tanya_informasi',
    'chitchat'
]

# NER labels
NER_LABELS = [
    'BAHAN_UTAMA',
    'BAHAN_TAMBAHAN',
    'BAHAN_HINDARI',
    'TEKNIK_MASAK',
    'KONDISI_KESEHATAN',
    'PREFERENSI_RASA',
    'WAKTU',
    'PORSI'
]

# Model parameters
TFIDF_PARAMS = {
    'max_features': 5000,
    'ngram_range': (1, 3),
    'min_df': 1,
    'max_df': 0.9
}

CLASSIFIER_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'random_state': 42
}

# NLP parameters
MIN_CONFIDENCE_THRESHOLD = 0.3
MAX_NGRAM_LENGTH = 4

# Output format
OUTPUT_FORMAT = {
    'version': '1.0',
    'timestamp': None,
    'user_input': None,
    'normalized_input': None,
    'intent': {
        'primary': None,
        'confidence': None,
        'secondary': []
    },
    'entities': {
        'ingredients': {
            'main': [],
            'additional': [],
            'avoid': []
        },
        'cooking_methods': [],
        'health_conditions': [],
        'taste_preferences': [],
        'time_constraint': None,
        'servings': None
    },
    'constraints': {
        'must_include': [],
        'must_exclude': [],
        'dietary_restrictions': []
    },
    'metadata': {
        'processing_time': None,
        'confidence_scores': {}
    }
}
