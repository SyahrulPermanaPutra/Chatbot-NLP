# train_model.py
# Script training intent classifier – kala_rasa_jtv NLP Service

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.intent_classifier import IntentClassifier


def main():
    print("=" * 70)
    print("  KALA RASA NLP – MODEL TRAINING")
    print("=" * 70)
    
    # Validasi direktori
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Error: Direktori '{data_dir}' tidak ditemukan!")
        print("   Pastikan Anda menjalankan script dari root project KalaRasa")
        return
    
    # Cek file-file penting
    required_files = [
        "data/Intents.json",
        "data/ner.json",
        "data/informal_map.json",
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"⚠ Warning: {f} tidak ditemukan")
    
    model_dir = os.getenv("MODEL_DIR", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Path ke Intents.json
    intents_json_path = os.path.join(data_dir, "Intents.json")
    use_intents_json = os.path.isfile(intents_json_path)
    
    clf = IntentClassifier()
    
    if use_intents_json:
        print(f"\n[1/3] Loading intents dari {intents_json_path} ...")
        results = clf.train_from_intents_json(intents_json_path)
    else:
        print(f"\n[1/3] File tidak ditemukan: {intents_json_path}")
        print("    Menggunakan built-in dataset minimal...")
        results = clf.train_from_builtin(augment=True)

    if results:
        print(f"\n[2/3] Menyimpan model ke {model_dir}/ ...")
        clf.save_model(model_dir)

        print("\n" + "=" * 70)
        print("  TRAINING SELESAI")
        print("=" * 70)
        print(f"  Train Accuracy : {results.get('train_score', 0):.4f}")
        print(f"  Test  Accuracy : {results.get('test_score', 0):.4f}")
        print(f"\n  Model tersimpan di:")
        print(f"    {model_dir}/intent_classifier.pkl")
        print(f"    {model_dir}/tfidf_vectorizer.pkl")

        # Smoke test predictions
        print("\n[3/3] Smoke test prediksi...")
        tests = [
            "mau masak ayam goreng",
            "aku diabetes carikan resep",
            "masakan padang yang enak",
            "halo",
        ]
        for t in tests:
            try:
                pred = clf.predict(t)
                print(f"  '{t}'")
                print(f"    → {pred['primary']} (conf: {pred['confidence']:.2f})")
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    main()