# API Documentation - Recipe NLP Chatbot

## Overview

Recipe NLP Chatbot menyediakan Python API untuk processing natural language queries tentang resep makanan.

---

## Core Classes

### 1. TextPreprocessor

**Location**: `src/preprocessor.py`

**Purpose**: Normalisasi dan cleaning input text

**Methods**:

#### `normalize_text(text: str) -> str`

Normalisasi text (lowercase, remove special chars, fix informal words)

```python
from src.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()
normalized = preprocessor.normalize_text("gw mau masak aym gorng")
# Output: "saya ingin masak ayam goreng"
```

#### `extract_negations(text: str) -> List[str]`

Extract pola negasi dari text

```python
negations = preprocessor.extract_negations("ga boleh pake santan")
# Output: ["ga boleh pake"]
```

#### `preprocess(text: str) -> Dict`

Main preprocessing function, returns full result

```python
result = preprocessor.preprocess("gw pengen masak ayam tanpa santan")
# Returns:
# {
#   'original': 'gw pengen masak ayam tanpa santan',
#   'normalized': 'saya ingin masak ayam tanpa santan',
#   'negations': ['tanpa santan']
# }
```

---

### 2. IntentClassifier

**Location**: `src/intent_classifier.py`

**Purpose**: Klasifikasi intent dari user query

**Methods**:

#### `load_model(vectorizer_path, classifier_path)`

Load pre-trained model

```python
from src.intent_classifier import IntentClassifier

classifier = IntentClassifier()
classifier.load_model()
```

#### `train(df: pd.DataFrame, test_size: float = 0.2)`

Train model dengan dataset

```python
import pandas as pd

df = pd.read_csv('data/intent_dataset.csv')
results = classifier.train(df, test_size=0.2)

print(f"Accuracy: {results['test_score']}")
```

#### `predict(text: str, top_k: int = 3) -> Dict`

Predict intent dari text

```python
result = classifier.predict("mau masak ayam goreng", top_k=3)
# Returns:
# {
#   'primary': 'cari_resep',
#   'confidence': 0.85,
#   'alternatives': [
#     {'intent': 'cari_resep_kompleks', 'confidence': 0.10},
#     {'intent': 'informasi_preferensi', 'confidence': 0.05}
#   ]
# }
```

**Parameters**:
- `text` (str): User input text
- `top_k` (int): Number of top predictions to return

**Returns**:
- `Dict` with primary intent, confidence, and alternatives

#### `save_model(vectorizer_path, classifier_path)`

Save trained model to disk

```python
classifier.save_model()
```

---

### 3. NERExtractor

**Location**: `src/ner_extractor.py`

**Purpose**: Extract entities dari text

**Methods**:

#### `extract_ingredients(text: str) -> Dict[str, List[str]]`

Extract ingredients dari text

```python
from src.ner_extractor import NERExtractor

ner = NERExtractor()
ingredients = ner.extract_ingredients("masak ayam tanpa santan")
# Returns:
# {
#   'main': ['ayam'],
#   'additional': [],
#   'avoid': ['santan']
# }
```

#### `extract_cooking_methods(text: str) -> List[str]`

Extract cooking methods

```python
methods = ner.extract_cooking_methods("mau goreng ayam yang crispy")
# Returns: ['goreng']
```

#### `extract_health_conditions(text: str) -> List[Dict]`

Extract health conditions dan pantangannya

```python
conditions = ner.extract_health_conditions("aku diabetes ga boleh gula")
# Returns:
# [
#   {
#     'name': 'diabetes',
#     'avoid': ['gula', 'nasi putih', ...],
#     'recommended': ['beras merah', 'sayuran hijau', ...]
#   }
# ]
```

#### `extract_taste_preferences(text: str) -> List[str]`

Extract taste preferences

```python
tastes = ner.extract_taste_preferences("mau yang pedas gurih")
# Returns: ['pedas', 'gurih']
```

#### `extract_time_constraint(text: str) -> str`

Extract time constraints jika ada

```python
time = ner.extract_time_constraint("yang cepat aja 15 menit")
# Returns: "15 minutes" atau "quick"
```

#### `extract_all(text: str) -> Dict`

Extract all entities sekaligus

```python
entities = ner.extract_all("mau masak ayam goreng pedas tanpa santan")
# Returns:
# {
#   'ingredients': {
#     'main': ['ayam'],
#     'additional': [],
#     'avoid': ['santan']
#   },
#   'cooking_methods': ['goreng'],
#   'health_conditions': [],
#   'taste_preferences': ['pedas'],
#   'time_constraint': None
# }
```

---

### 4. RecipeNLPPipeline

**Location**: `src/nlp_pipeline.py`

**Purpose**: Main pipeline yang integrate semua komponen

**Methods**:

#### `__init__(load_models: bool = True)`

Initialize pipeline

```python
from src.nlp_pipeline import RecipeNLPPipeline

# With pre-trained models
pipeline = RecipeNLPPipeline(load_models=True)

# Without models (for training first)
pipeline = RecipeNLPPipeline(load_models=False)
```

#### `process(user_input: str) -> Dict`

Process user input through full pipeline

```python
result = pipeline.process("gw pengen masak ayam goreng tanpa tepung")
```

**Returns**: Complete structured JSON output

```json
{
  "version": "1.0",
  "timestamp": "2024-02-05T10:30:00",
  "user_input": "gw pengen masak ayam goreng tanpa tepung",
  "normalized_input": "saya ingin masak ayam goreng tanpa tepung",
  "intent": {
    "primary": "cari_resep_kompleks",
    "confidence": 0.87,
    "secondary": [...]
  },
  "entities": {
    "ingredients": {
      "main": ["ayam"],
      "additional": [],
      "avoid": ["tepung"]
    },
    "cooking_methods": ["goreng"],
    "health_conditions": [],
    "taste_preferences": [],
    "time_constraint": null,
    "servings": null
  },
  "constraints": {
    "must_include": ["ayam"],
    "must_exclude": ["tepung"],
    "dietary_restrictions": []
  },
  "metadata": {
    "processing_time": 0.0234,
    "confidence_scores": {
      "intent": 0.87
    }
  }
}
```

#### `process_batch(inputs: List[str]) -> List[Dict]`

Process multiple inputs sekaligus

```python
inputs = [
    "mau masak ayam",
    "aku diabetes",
    "pengen yang pedas"
]

results = pipeline.process_batch(inputs)
# Returns: List of result dictionaries
```

#### `get_summary(output: Dict) -> str`

Generate human-readable summary

```python
result = pipeline.process("mau masak ayam goreng")
summary = pipeline.get_summary(result)
print(summary)
# Output:
# Intent: cari_resep (85%)
# Main ingredients: ayam
# Cooking methods: goreng
```

---

## Output Schema

### Complete Output Structure

```json
{
  "version": "string",              // API version
  "timestamp": "ISO8601",           // Processing timestamp
  "user_input": "string",           // Original input
  "normalized_input": "string",     // Processed input
  
  "intent": {
    "primary": "string",            // Main intent
    "confidence": "float",          // 0-1 confidence score
    "secondary": [                  // Alternative intents
      {
        "intent": "string",
        "confidence": "float"
      }
    ]
  },
  
  "entities": {
    "ingredients": {
      "main": ["string"],           // Main ingredients
      "additional": ["string"],     // Additional ingredients
      "avoid": ["string"]           // Ingredients to avoid
    },
    "cooking_methods": ["string"],  // Cooking techniques
    "health_conditions": ["string"], // Health conditions
    "taste_preferences": ["string"], // Taste preferences
    "time_constraint": "string",    // Time constraints
    "servings": "int"               // Number of servings
  },
  
  "constraints": {
    "must_include": ["string"],     // Required ingredients
    "must_exclude": ["string"],     // Forbidden ingredients
    "dietary_restrictions": [       // Health-based restrictions
      {
        "condition": "string",
        "avoid": ["string"],
        "recommended": ["string"]
      }
    ]
  },
  
  "metadata": {
    "processing_time": "float",     // Time in seconds
    "confidence_scores": {
      "intent": "float"
    }
  }
}
```

---

## Intent Types

| Intent | Description | Example Query |
|--------|-------------|---------------|
| `cari_resep` | Basic recipe search | "mau masak ayam goreng" |
| `cari_resep_kondisi` | Recipe for health condition | "resep untuk diabetes" |
| `cari_resep_pantangan` | Recipe with restrictions | "masak tanpa santan" |
| `cari_resep_kompleks` | Complex multi-constraint | "ayam goreng tanpa tepung untuk diet" |
| `informasi_kondisi_kesehatan` | User mentions health condition | "aku diabetes" |
| `informasi_pantangan` | User mentions restriction | "ga bisa makan pedas" |
| `informasi_preferensi` | User mentions preference | "mau yang gurih" |
| `tanya_alternatif` | Ask for substitution | "santan bisa diganti apa?" |
| `tanya_informasi` | Ask for info | "cara masak nasi yang pulen?" |
| `chitchat` | General conversation | "terima kasih" |

---

## Entity Types

| Entity | Description | Example Values |
|--------|-------------|----------------|
| `BAHAN_UTAMA` | Main ingredients | ayam, ikan, tempe |
| `BAHAN_TAMBAHAN` | Additional ingredients | bawang, cabai |
| `BAHAN_HINDARI` | Ingredients to avoid | santan, gula |
| `TEKNIK_MASAK` | Cooking methods | goreng, rebus, tumis |
| `KONDISI_KESEHATAN` | Health conditions | diabetes, kolesterol |
| `PREFERENSI_RASA` | Taste preferences | pedas, manis, gurih |
| `WAKTU` | Time constraints | "15 menit", "cepat" |
| `PORSI` | Servings | "4 orang" |

---

## Error Handling

### Exception: ModelNotTrainedError

Raised when trying to predict without training/loading model

```python
try:
    classifier = IntentClassifier()
    result = classifier.predict("test")
except ValueError as e:
    print("Model not trained! Run train_model.py first")
```

### Exception: FileNotFoundError

Raised when knowledge base files not found

```python
try:
    ner = NERExtractor()
except FileNotFoundError as e:
    print(f"Knowledge base file missing: {e}")
```

---

## Configuration

All configurations in `config/config.py`:

```python
from config.config import *

# Paths
print(KB_INGREDIENTS)        # Knowledge base paths
print(INTENT_MODEL)          # Model paths

# Parameters
print(TFIDF_PARAMS)          # TF-IDF parameters
print(CLASSIFIER_PARAMS)     # Classifier parameters
print(MIN_CONFIDENCE_THRESHOLD)  # Minimum confidence

# Labels
print(INTENT_LABELS)         # All intent types
print(NER_LABELS)            # All entity types
```

---

## Performance

Typical performance metrics:

| Metric | Value |
|--------|-------|
| Processing time | 20-50ms per query |
| Intent accuracy | 85-95% (with good data) |
| NER recall | 70-80% (rule-based) |
| Throughput | ~20-50 queries/second |

---

## Best Practices

### 1. Always Normalize Input

```python
# Good
preprocessor = TextPreprocessor()
normalized = preprocessor.normalize_text(user_input)
result = pipeline.process(normalized)

# Also good (pipeline does it internally)
result = pipeline.process(user_input)
```

### 2. Check Confidence Scores

```python
result = pipeline.process(query)

if result['intent']['confidence'] < 0.5:
    # Low confidence - ask user to clarify
    print("I'm not sure what you mean. Can you rephrase?")
else:
    # High confidence - proceed
    process_recipe_search(result)
```

### 3. Handle Missing Entities

```python
result = pipeline.process(query)

if not result['entities']['ingredients']['main']:
    # No main ingredient detected
    print("What ingredient would you like to use?")
```

### 4. Use Batch Processing for Multiple Queries

```python
# Good for processing many queries
queries = get_user_queries()  # List of queries
results = pipeline.process_batch(queries)

# Better than looping individual process()
```

---

## Examples

See `demo.py` for comprehensive examples or run:

```bash
python demo.py
```

---

## Changelog

### Version 1.0 (Current)
- Initial release
- Intent classification
- Rule-based NER
- Knowledge base system
- Batch processing support

### Planned for 2.0
- Model-based NER
- Context memory
- API endpoints
- Database integration

---

## Support

For issues or questions:
1. Check `README.md` for overview
2. Check `QUICKSTART.md` for quick setup
3. Run `python test_all.py` for diagnostics
4. Run `python demo.py` for examples
