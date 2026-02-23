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

    model_dir = os.getenv("MODEL_DIR", "models")
    os.makedirs(model_dir, exist_ok=True)

    # Cek apakah ada CSV eksternal
    csv_path = os.getenv("INTENT_DATASET", "data/intent_dataset.csv")
    use_csv = os.path.isfile(csv_path)

    clf = IntentClassifier()

    if use_csv:
        print(f"\n[1/3] Loading dataset dari {csv_path} ...")
        results = clf.train_from_csv(csv_path)
    else:
        print("\n[1/3] Menggunakan built-in dataset (tidak ada CSV ditemukan) ...")
        results = clf.train_from_builtin(augment=True)

    print(f"\n[2/3] Menyimpan model ke {model_dir}/ ...")
    clf.save_model(model_dir)

    print("\n" + "=" * 70)
    print("  TRAINING SELESAI")
    print("=" * 70)
    print(f"  Train Accuracy : {results['train_score']:.4f}")
    print(f"  Test  Accuracy : {results['test_score']:.4f}")
    print(f"\n  Model tersimpan di:")
    print(f"    {model_dir}/intent_classifier.pkl")
    print(f"    {model_dir}/tfidf_vectorizer.pkl")

    # Smoke test predictions
    print("\n[3/3] Smoke test prediksi...")
    tests = [
        "mau masak ayam goreng",
        "aku diabetes carikan resep",
        "masakan padang yang enak",
        "lihat detail resep",
        "simpan ke favorit",
        "halo",
    ]
    for t in tests:
        pred = clf.predict(t)
        print(f"  '{t}'")
        print(f"    → {pred['primary']} (conf: {pred['confidence']:.2f})")


if __name__ == "__main__":
    main()