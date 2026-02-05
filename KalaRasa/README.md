# Recipe NLP Chatbot

Sistem NLP untuk memahami permintaan resep makanan dalam bahasa Indonesia dengan fokus pada ekstraksi informasi terstruktur.

## 📋 Overview

Project ini adalah **tahap awal** sistem rekomendasi resep yang fokus pada:
- ✅ Memahami input teks natural language dari pengguna
- ✅ Mengekstraksi informasi penting (intent, entities)
- ✅ Menghasilkan output JSON terstruktur
- ❌ **BUKAN** sistem rekomendasi (tahap selanjutnya)
- ❌ **BUKAN** menggunakan deep learning (MVP sederhana)

## 🎯 Tujuan Project

1. **Intent Classification**: Menentukan maksud utama user
2. **Entity Extraction (NER)**: Mengekstrak bahan, teknik masak, pantangan, dll
3. **Knowledge Base**: Database statis untuk mapping kondisi kesehatan → pantangan
4. **Structured Output**: JSON yang siap digunakan recipe matcher

## 🏗️ Arsitektur Sistem

```
User Input (Raw Text)
        ↓
[1] Text Preprocessing
    - Normalisasi kata informal (gw → saya, gak → tidak)
    - Koreksi typo umum
    - Cleaning
        ↓
[2] Intent Classification
    - TF-IDF Vectorization
    - Random Forest Classifier
    - Confidence scoring
        ↓
[3] Named Entity Recognition (NER)
    - Rule-based + Pattern matching
    - Knowledge base lookup
    - N-gram extraction
        ↓
[4] Structured JSON Output
    - Intent + confidence
    - Extracted entities
    - Constraints & preferences
        ↓
Recipe Matcher (Future work)
```

## 📊 Intent Labels

Project ini menangani 10 intent utama:

1. **cari_resep**: Mencari resep sederhana
   - "mau masak ayam goreng"
   
2. **cari_resep_kondisi**: Cari resep untuk kondisi tertentu
   - "resep untuk diabetes"
   
3. **cari_resep_pantangan**: Cari resep dengan pantangan
   - "masak tanpa santan"
   
4. **cari_resep_kompleks**: Kombinasi multiple constraints
   - "ayam goreng tanpa tepung untuk diet"
   
5. **informasi_kondisi_kesehatan**: User menyebutkan kondisi kesehatan
   - "aku diabetes"
   
6. **informasi_pantangan**: User menyebutkan pantangan
   - "ga bisa makan pedas"
   
7. **informasi_preferensi**: User menyebutkan preferensi
   - "mau yang gurih"
   
8. **tanya_alternatif**: Bertanya substitusi bahan
   - "santan bisa diganti apa?"
   
9. **tanya_informasi**: Bertanya informasi umum
   - "cara masak nasi yang pulen gimana?"
   
10. **chitchat**: Obrolan umum
    - "terima kasih", "halo"

## 🏷️ Entity Labels (NER)

Sistem mengekstrak 8 jenis entity:

1. **BAHAN_UTAMA**: Bahan makanan utama
   - ayam, ikan, tempe, sayur
   
2. **BAHAN_TAMBAHAN**: Bahan pelengkap
   - bawang putih, cabai, garam
   
3. **BAHAN_HINDARI**: Bahan yang harus dihindari
   - "tanpa santan", "ga pakai gula"
   
4. **TEKNIK_MASAK**: Metode memasak
   - goreng, rebus, panggang, tumis
   
5. **KONDISI_KESEHATAN**: Kondisi kesehatan user
   - diabetes, kolesterol, asam urat
   
6. **PREFERENSI_RASA**: Preferensi rasa
   - pedas, manis, gurih, segar
   
7. **WAKTU**: Batasan waktu
   - "15 menit", "cepat", "simple"
   
8. **PORSI**: Jumlah porsi (future)

## 📁 Struktur Project

```
recipe_nlp_chatbot/
├── config/
│   └── config.py                    # Konfigurasi sistem
├── data/
│   ├── knowledge_base_ingredients.json      # Database bahan makanan
│   ├── knowledge_base_cooking_methods.json  # Database teknik masak
│   ├── knowledge_base_health_conditions.json # Database kondisi kesehatan
│   ├── knowledge_base_normalization.json    # Dictionary normalisasi
│   ├── intent_dataset.csv                   # Dataset training intent
│   ├── ner_dataset.csv                      # Dataset training NER
│   └── recipe_database.json                 # Dummy recipe database
├── src/
│   ├── preprocessor.py              # Text preprocessing
│   ├── intent_classifier.py         # Intent classification
│   ├── ner_extractor.py            # Named entity recognition
│   └── nlp_pipeline.py             # Main pipeline integration
├── models/                          # Trained models (generated)
│   ├── intent_classifier.pkl
│   └── tfidf_vectorizer.pkl
├── outputs/                         # Output JSON files
├── train_model.py                   # Training script
├── chatbot.py                       # Interactive chatbot
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone atau extract project
cd recipe_nlp_chatbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Training Model

```bash
# Train intent classifier
python train_model.py
```

Output:
```
=== RECIPE NLP CHATBOT - MODEL TRAINING ===
[1/4] Initializing intent classifier...
[2/4] Loading dataset...
      Loaded 100 training examples
[3/4] Training model...
      Train Accuracy: 0.9875
      Test Accuracy: 0.8500
[4/4] Saving model...
✓ Training completed!
```

### 3. Running Chatbot

```bash
# Run interactive chatbot
python chatbot.py
```

### 4. Testing Components

```bash
# Test preprocessor
python src/preprocessor.py

# Test intent classifier
python src/intent_classifier.py

# Test NER extractor
python src/ner_extractor.py

# Test full pipeline
python src/nlp_pipeline.py
```

## 💡 Contoh Penggunaan

### Interactive Chatbot

```
You: gw pengen masak ayam goreng yang krispy tapi gak pake tepung

Bot:
Saya mengerti!
Intent terdeteksi: cari_resep_kompleks (confidence: 87.5%)

Bahan utama: ayam
Hindari: tepung
Teknik memasak: goreng
Preferensi rasa: krispy

✓ Informasi berhasil diekstrak!
Data ini siap dikirim ke sistem recipe matcher.
```

### Programmatic Usage

```python
from src.nlp_pipeline import RecipeNLPPipeline
import json

# Initialize pipeline
pipeline = RecipeNLPPipeline(load_models=True)

# Process input
user_input = "mau masak ikan bakar yang cepat untuk diabetes"
result = pipeline.process(user_input)

# Print JSON output
print(json.dumps(result, indent=2, ensure_ascii=False))
```

Output JSON:
```json
{
  "version": "1.0",
  "timestamp": "2024-02-05T10:30:00",
  "user_input": "mau masak ikan bakar yang cepat untuk diabetes",
  "normalized_input": "ingin masak ikan bakar yang cepat untuk diabetes",
  "intent": {
    "primary": "cari_resep_kondisi",
    "confidence": 0.92,
    "secondary": [
      {"intent": "cari_resep", "confidence": 0.05}
    ]
  },
  "entities": {
    "ingredients": {
      "main": ["ikan"],
      "additional": [],
      "avoid": ["gula", "nasi putih", "tepung terigu"]
    },
    "cooking_methods": ["bakar"],
    "health_conditions": ["diabetes"],
    "taste_preferences": [],
    "time_constraint": "quick"
  },
  "constraints": {
    "must_include": ["ikan"],
    "must_exclude": ["gula", "nasi putih"],
    "dietary_restrictions": [
      {
        "condition": "diabetes",
        "avoid": ["gula", "nasi putih", "tepung terigu"],
        "recommended": ["beras merah", "sayuran hijau"]
      }
    ]
  },
  "metadata": {
    "processing_time": 0.0234,
    "confidence_scores": {
      "intent": 0.92
    }
  }
}
```

## 🔧 Customization

### Menambah Intent Baru

1. Tambahkan intent ke `config/config.py`:
```python
INTENT_LABELS = [
    'cari_resep',
    'intent_baru_anda',  # Tambahkan di sini
    ...
]
```

2. Tambahkan training data ke `data/intent_dataset.csv`:
```csv
text,intent
"contoh kalimat untuk intent baru",intent_baru_anda
```

3. Re-train model:
```bash
python train_model.py
```

### Menambah Bahan Makanan Baru

Edit `data/knowledge_base_ingredients.json`:
```json
{
  "protein": {
    "daging": ["ayam", "sapi", "bahan_baru_anda"]
  }
}
```

### Menambah Kondisi Kesehatan Baru

Edit `data/knowledge_base_health_conditions.json`:
```json
{
  "kondisi_kesehatan": {
    "kondisi_baru": {
      "nama": "kondisi baru",
      "sinonim": ["sinonim1", "sinonim2"],
      "hindari": ["bahan1", "bahan2"],
      "anjuran": ["bahan3", "bahan4"]
    }
  }
}
```

## 📊 Model Performance

Berdasarkan training dengan dataset dummy:

| Metric | Score |
|--------|-------|
| Training Accuracy | ~98% |
| Testing Accuracy | ~85% |
| Processing Time | <50ms per query |

**Note**: Performance akan meningkat dengan dataset yang lebih besar dan beragam.

## 🔮 Future Work

Tahap selanjutnya yang BELUM diimplementasikan:

1. **Recipe Matcher**: Sistem untuk matching extracted info dengan database resep
2. **Ranking System**: Algoritma untuk ranking resep berdasarkan kesesuaian
3. **Deep Learning NER**: Upgrade dari rule-based ke model-based NER
4. **Conversational Context**: Memory untuk multi-turn conversation
5. **Recipe Database Integration**: Koneksi ke database resep yang sebenarnya
6. **API Development**: REST API untuk integrasi dengan aplikasi lain

## 🐛 Known Limitations

1. **Rule-based NER**: Tergantung pada knowledge base, bisa miss entities baru
2. **No Context Memory**: Setiap input diproses independent
3. **Indonesian Only**: Belum optimal untuk mixed language
4. **Limited Dataset**: Perlu lebih banyak training data untuk generalisasi better
5. **No Spell Correction**: Hanya normalisasi basic, belum advanced spell checking

## 📝 Design Decisions

### Kenapa Rule-based NER?
- ✅ Cepat untuk MVP
- ✅ Mudah debug dan maintain
- ✅ Tidak perlu banyak labeled data
- ✅ Transparent dan explainable
- ❌ Tapi: Limited scalability

### Kenapa TF-IDF + Random Forest?
- ✅ Simple dan cepat train
- ✅ Good baseline performance
- ✅ Tidak perlu GPU
- ✅ Interpretable features
- ❌ Tapi: Tidak secangih transformer models

### Kenapa Tidak Deep Learning?
- Prioritas: **Kestabilan struktur > Kecanggihan model**
- MVP harus cepat delivered
- Dataset masih kecil
- Infrastructure requirements rendah

## 🤝 Contributing

Untuk berkontribusi:

1. Tambah training data ke `data/intent_dataset.csv` atau `data/ner_dataset.csv`
2. Perbaiki knowledge base di `data/knowledge_base_*.json`
3. Report bugs atau request features
4. Submit test cases untuk edge cases

## 📄 License

MIT License - Feel free to use and modify

## 👥 Authors

Recipe NLP Chatbot Project - 2024

---

**Status**: ✅ MVP Ready - Phase 1 Complete

**Next Phase**: Recipe Matching & Recommendation System
