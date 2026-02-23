# src/intent_classifier.py
# Intent Classification – TF-IDF + Random Forest
# Intent diselaraskan dengan kolom user_queries.intent di database kala_rasa_jtv
#
# Intent yang tersedia (sesuai user_queries.intent):
#   cari_resep           – mencari resep berdasarkan bahan/nama
#   cari_resep_sehat     – mencari resep dengan filter kondisi kesehatan
#   filter_bahan         – menambah/mengubah filter bahan
#   filter_waktu         – filter berdasarkan waktu masak
#   filter_region        – filter berdasarkan daerah asal masakan
#   lihat_detail         – meminta detail resep tertentu
#   tanya_pantangan      – bertanya pantangan/pembatasan makanan
#   tanya_nutrisi        – bertanya kandungan gizi
#   tambah_favorit       – menambahkan resep ke favorit
#   hapus_favorit        – menghapus resep dari favorit
#   lihat_favorit        – melihat daftar favorit
#   chitchat             – percakapan umum
#   unknown              – tidak dikenali

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.preprocessor import TextPreprocessor


class IntentClassifier:
    """
    Classifier intent berbasis TF-IDF + Random Forest.
    Menggunakan dataset built-in sehingga tidak bergantung file CSV eksternal.
    """

    # ----------------------------------------------------------------
    # Threshold
    # ----------------------------------------------------------------
    MIN_CONFIDENCE = 0.35

    # ----------------------------------------------------------------
    # Dataset built-in (training examples)
    # ----------------------------------------------------------------
    TRAINING_DATA: List[Dict] = [
        # ── cari_resep ──────────────────────────────────────────────
        {"text": "mau masak ayam goreng", "intent": "cari_resep"},
        {"text": "ingin bikin ikan bakar", "intent": "cari_resep"},
        {"text": "carikan resep nasi goreng", "intent": "cari_resep"},
        {"text": "resep soto ayam dong", "intent": "cari_resep"},
        {"text": "pengen bikin tempe mendoan", "intent": "cari_resep"},
        {"text": "cari resep rendang sapi", "intent": "cari_resep"},
        {"text": "mau masak mie goreng", "intent": "cari_resep"},
        {"text": "ingin buat sate kambing", "intent": "cari_resep"},
        {"text": "carikan resep opor ayam", "intent": "cari_resep"},
        {"text": "resep gado-gado apa ya", "intent": "cari_resep"},
        {"text": "mau masak bakso sapi", "intent": "cari_resep"},
        {"text": "ingin buat sop ayam", "intent": "cari_resep"},
        {"text": "carikan resep gulai ikan", "intent": "cari_resep"},
        {"text": "pengen masak cap cay", "intent": "cari_resep"},
        {"text": "resep capcay udang dong", "intent": "cari_resep"},
        {"text": "bikin sayur asem yuk", "intent": "cari_resep"},
        {"text": "mau masak dengan ayam", "intent": "cari_resep"},
        {"text": "ada bahan udang mau masak apa", "intent": "cari_resep"},
        {"text": "punya tempe mau diapain", "intent": "cari_resep"},
        {"text": "masak apa pakai daging sapi", "intent": "cari_resep"},
        {"text": "resep pasta carbonara", "intent": "cari_resep"},
        {"text": "mie goreng jawa", "intent": "cari_resep"},
        {"text": "resep nasi uduk", "intent": "cari_resep"},
        {"text": "mau buat pecel lele", "intent": "cari_resep"},
        {"text": "cari resep pindang ikan", "intent": "cari_resep"},
        {"text": "ingin masak tumis kangkung", "intent": "cari_resep"},
        {"text": "resep apa yang bisa dibuat dari telur", "intent": "cari_resep"},
        {"text": "mau masak tahu goreng", "intent": "cari_resep"},
        {"text": "resep kwetiau goreng", "intent": "cari_resep"},
        {"text": "bikin nasi kuning dong", "intent": "cari_resep"},

        # ── cari_resep_sehat ─────────────────────────────────────────
        {"text": "aku diabetes mau masak apa", "intent": "cari_resep_sehat"},
        {"text": "punya kolesterol tinggi carikan resep", "intent": "cari_resep_sehat"},
        {"text": "mau masak yang cocok untuk hipertensi", "intent": "cari_resep_sehat"},
        {"text": "resep untuk penderita asam urat", "intent": "cari_resep_sehat"},
        {"text": "mau diet carikan resep sehat", "intent": "cari_resep_sehat"},
        {"text": "aku vegetarian mau masak apa", "intent": "cari_resep_sehat"},
        {"text": "alergi gluten ada resep apa", "intent": "cari_resep_sehat"},
        {"text": "dairy free carikan resep", "intent": "cari_resep_sehat"},
        {"text": "vegan mau masak apa ya", "intent": "cari_resep_sehat"},
        {"text": "sakit maag bisa makan apa", "intent": "cari_resep_sehat"},
        {"text": "penyakit jantung makanan apa yang aman", "intent": "cari_resep_sehat"},
        {"text": "kurang darah anemia makan apa baiknya", "intent": "cari_resep_sehat"},
        {"text": "obesitas mau masak yang rendah kalori", "intent": "cari_resep_sehat"},
        {"text": "resep sehat untuk diabetes tanpa gula", "intent": "cari_resep_sehat"},
        {"text": "masakan untuk diet ketat", "intent": "cari_resep_sehat"},
        {"text": "resep ayam untuk penderita kolesterol", "intent": "cari_resep_sehat"},
        {"text": "ada resep tidak pake santan untuk darah tinggi", "intent": "cari_resep_sehat"},
        {"text": "masakan rendah sodium untuk hipertensi", "intent": "cari_resep_sehat"},

        # ── filter_bahan ─────────────────────────────────────────────
        {"text": "yang tidak pakai santan", "intent": "filter_bahan"},
        {"text": "tanpa tepung bisa tidak", "intent": "filter_bahan"},
        {"text": "ganti ayam dengan tahu", "intent": "filter_bahan"},
        {"text": "tanpa bawang putih boleh", "intent": "filter_bahan"},
        {"text": "tidak pakai telur", "intent": "filter_bahan"},
        {"text": "yang bebas gluten", "intent": "filter_bahan"},
        {"text": "mau yang tidak pedas", "intent": "filter_bahan"},
        {"text": "ganti dagingnya dengan tempe", "intent": "filter_bahan"},
        {"text": "kalau tidak ada udang pakai apa", "intent": "filter_bahan"},
        {"text": "tidak mau pakai minyak banyak", "intent": "filter_bahan"},
        {"text": "kurangi garamnya bisa", "intent": "filter_bahan"},
        {"text": "alternatif bahan selain ikan", "intent": "filter_bahan"},

        # ── filter_waktu ──────────────────────────────────────────────
        {"text": "yang cepat dibuat 30 menit", "intent": "filter_waktu"},
        {"text": "mau yang simpel dan cepat", "intent": "filter_waktu"},
        {"text": "resep yang mudah dan tidak lama", "intent": "filter_waktu"},
        {"text": "yang kurang dari 1 jam", "intent": "filter_waktu"},
        {"text": "masak cepat 15 menit", "intent": "filter_waktu"},
        {"text": "resep kilat untuk makan siang", "intent": "filter_waktu"},
        {"text": "yang gampang dan tidak ribet", "intent": "filter_waktu"},
        {"text": "mau yang 20 menit sudah jadi", "intent": "filter_waktu"},

        # ── filter_region ─────────────────────────────────────────────
        {"text": "masakan padang enak apa", "intent": "filter_region"},
        {"text": "resep masakan jawa tradisional", "intent": "filter_region"},
        {"text": "masakan bali yang otentik", "intent": "filter_region"},
        {"text": "ada resep masakan sunda", "intent": "filter_region"},
        {"text": "resep dari manado", "intent": "filter_region"},
        {"text": "masakan betawi apa saja", "intent": "filter_region"},
        {"text": "resep jepang yang bisa dibuat di rumah", "intent": "filter_region"},
        {"text": "masakan korea yang populer", "intent": "filter_region"},
        {"text": "resep italia untuk makan malam", "intent": "filter_region"},
        {"text": "kuliner aceh yang terkenal", "intent": "filter_region"},

        # ── lihat_detail ──────────────────────────────────────────────
        {"text": "lihat resep nomor 1", "intent": "lihat_detail"},
        {"text": "tampilkan resep yang pertama", "intent": "lihat_detail"},
        {"text": "buka resep rendang itu", "intent": "lihat_detail"},
        {"text": "detail resep soto ayam", "intent": "lihat_detail"},
        {"text": "cara membuat rendang lengkap", "intent": "lihat_detail"},
        {"text": "langkah langkah masak nasi goreng", "intent": "lihat_detail"},
        {"text": "bahan apa saja untuk sate ayam", "intent": "lihat_detail"},
        {"text": "tampilkan yang kedua", "intent": "lihat_detail"},
        {"text": "resep ketiga mau lihat", "intent": "lihat_detail"},
        {"text": "boleh dilihat resep itu", "intent": "lihat_detail"},

        # ── tanya_pantangan ──────────────────────────────────────────
        {"text": "apa yang tidak boleh dimakan penderita diabetes", "intent": "tanya_pantangan"},
        {"text": "pantangan makanan untuk hipertensi apa saja", "intent": "tanya_pantangan"},
        {"text": "makanan yang harus dihindari kolesterol tinggi", "intent": "tanya_pantangan"},
        {"text": "asam urat tidak boleh makan apa", "intent": "tanya_pantangan"},
        {"text": "apa pantangan maag", "intent": "tanya_pantangan"},
        {"text": "makanan apa yang dilarang untuk vegan", "intent": "tanya_pantangan"},
        {"text": "penderita jantung tidak boleh makan apa", "intent": "tanya_pantangan"},
        {"text": "diet ketat apa saja yang dihindari", "intent": "tanya_pantangan"},
        {"text": "alergi gluten tidak boleh makan apa", "intent": "tanya_pantangan"},

        # ── tanya_nutrisi ─────────────────────────────────────────────
        {"text": "berapa kalori ayam goreng", "intent": "tanya_nutrisi"},
        {"text": "kandungan gizi tempe apa saja", "intent": "tanya_nutrisi"},
        {"text": "protein dalam telur berapa", "intent": "tanya_nutrisi"},
        {"text": "nasi goreng kalorinya berapa", "intent": "tanya_nutrisi"},
        {"text": "informasi nutrisi rendang", "intent": "tanya_nutrisi"},
        {"text": "lemak dalam santan berapa", "intent": "tanya_nutrisi"},
        {"text": "karbohidrat mie berapa gram", "intent": "tanya_nutrisi"},

        # ── tambah_favorit ─────────────────────────────────────────────
        {"text": "simpan resep ini ke favorit", "intent": "tambah_favorit"},
        {"text": "tambahkan ke daftar favorit saya", "intent": "tambah_favorit"},
        {"text": "suka resep ini mau disimpan", "intent": "tambah_favorit"},
        {"text": "bookmark resep rendang ini", "intent": "tambah_favorit"},
        {"text": "masukkan ke favorit", "intent": "tambah_favorit"},

        # ── hapus_favorit ─────────────────────────────────────────────
        {"text": "hapus dari favorit", "intent": "hapus_favorit"},
        {"text": "tidak suka resep ini hapus saja", "intent": "hapus_favorit"},
        {"text": "remove dari bookmark", "intent": "hapus_favorit"},
        {"text": "keluarkan dari daftar favorit", "intent": "hapus_favorit"},

        # ── lihat_favorit ─────────────────────────────────────────────
        {"text": "lihat daftar favorit saya", "intent": "lihat_favorit"},
        {"text": "tampilkan resep yang sudah disimpan", "intent": "lihat_favorit"},
        {"text": "resep favorit saya apa saja", "intent": "lihat_favorit"},
        {"text": "bookmark saya mana", "intent": "lihat_favorit"},
        {"text": "resep yang pernah saya simpan", "intent": "lihat_favorit"},

        # ── chitchat ─────────────────────────────────────────────────
        {"text": "halo", "intent": "chitchat"},
        {"text": "hai", "intent": "chitchat"},
        {"text": "selamat pagi", "intent": "chitchat"},
        {"text": "selamat siang", "intent": "chitchat"},
        {"text": "selamat malam", "intent": "chitchat"},
        {"text": "terima kasih", "intent": "chitchat"},
        {"text": "makasih ya", "intent": "chitchat"},
        {"text": "oke siap", "intent": "chitchat"},
        {"text": "baik terima kasih", "intent": "chitchat"},
        {"text": "kamu siapa", "intent": "chitchat"},
        {"text": "kamu bisa apa saja", "intent": "chitchat"},
        {"text": "sampai jumpa", "intent": "chitchat"},
        {"text": "dadah", "intent": "chitchat"},
        {"text": "bye bye", "intent": "chitchat"},
        {"text": "tolong bantu saya", "intent": "chitchat"},
        {"text": "sudah cukup terima kasih", "intent": "chitchat"},
    ]

    # ----------------------------------------------------------------
    # TF-IDF & classifier params
    # ----------------------------------------------------------------
    TFIDF_PARAMS = {
        "max_features": 3000,
        "ngram_range": (1, 3),
        "min_df": 1,
        "max_df": 0.95,
        "sublinear_tf": True,
    }

    CLASSIFIER_PARAMS = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[RandomForestClassifier] = None
        self._trained = False

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------

    def train_from_builtin(self, augment: bool = True) -> Dict:
        """Train menggunakan dataset built-in."""
        df = pd.DataFrame(self.TRAINING_DATA)
        if augment:
            df = self._augment_data(df)
        return self._train(df)

    def train_from_csv(self, filepath: str) -> Dict:
        """Train menggunakan file CSV eksternal (kolom: text, intent)."""
        df = pd.read_csv(filepath)
        df = self._augment_data(df)
        return self._train(df)

    def _train(self, df: pd.DataFrame) -> Dict:
        df["processed"] = df["text"].apply(self.preprocessor.normalize_text)

        X_train, X_test, y_train, y_test = train_test_split(
            df["processed"], df["intent"],
            test_size=0.2, random_state=42, stratify=df["intent"]
        )

        self.vectorizer = TfidfVectorizer(**self.TFIDF_PARAMS)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.classifier = RandomForestClassifier(**self.CLASSIFIER_PARAMS)
        self.classifier.fit(X_train_vec, y_train)
        self._trained = True

        train_acc = self.classifier.score(X_train_vec, y_train)
        test_acc = self.classifier.score(X_test_vec, y_test)
        y_pred = self.classifier.predict(X_test_vec)

        print(f"Train accuracy : {train_acc:.4f}")
        print(f"Test  accuracy : {test_acc:.4f}")
        print(classification_report(y_test, y_pred))

        return {"train_score": train_acc, "test_score": test_acc}

    def _augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data augmentation sederhana."""
        augmented = []
        synonym_map = {
            "mau": ["ingin", "pengen", "kepingin"],
            "masak": ["bikin", "buat", "membuat"],
            "resep": ["cara masak", "cara membuat"],
            "ayam": ["daging ayam"],
            "ikan": ["ikan segar"],
            "cepat": ["cepet", "kilat"],
        }
        for _, row in df.iterrows():
            augmented.append(row.to_dict())
            text, intent = row["text"], row["intent"]
            if intent != "chitchat":
                for word, syns in synonym_map.items():
                    if word in text:
                        augmented.append({"text": text.replace(word, syns[0]), "intent": intent})
        return pd.DataFrame(augmented)

    # ----------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------

    def predict(self, text: str, top_k: int = 3) -> Dict:
        """
        Prediksi intent.
        Returns: {primary, confidence, alternatives}
        """
        if not self._trained:
            raise RuntimeError("Model belum ditraining. Jalankan train_from_builtin() dulu.")

        processed = self.preprocessor.normalize_text(text)
        vec = self.vectorizer.transform([processed])
        proba = self.classifier.predict_proba(vec)[0]

        top_idx = np.argsort(proba)[::-1][:top_k]
        top_intents = self.classifier.classes_[top_idx]
        top_probas = proba[top_idx]

        return {
            "primary": top_intents[0],
            "confidence": float(top_probas[0]),
            "alternatives": [
                {"intent": i, "confidence": float(p)}
                for i, p in zip(top_intents[1:], top_probas[1:])
                if p > self.MIN_CONFIDENCE
            ],
        }

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def save_model(self, dir_path: str = "models") -> None:
        os.makedirs(dir_path, exist_ok=True)
        with open(f"{dir_path}/intent_classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)
        with open(f"{dir_path}/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"Model saved → {dir_path}/")

    def load_model(self, dir_path: str = "models") -> None:
        with open(f"{dir_path}/intent_classifier.pkl", "rb") as f:
            self.classifier = pickle.load(f)
        with open(f"{dir_path}/tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        self._trained = True
        print("Model loaded ✓")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    clf = IntentClassifier()
    clf.train_from_builtin()

    tests = [
        "mau masak ayam goreng yang crispy",
        "aku diabetes jadi ga boleh makan manis",
        "resep padang yang enak",
        "lihat resep nomor 2",
        "simpan ke favorit",
        "berapa kalori rendang",
        "halo selamat pagi",
        "masak apa yang cepat 20 menit",
    ]
    print("\n=== Prediction Test ===")
    for t in tests:
        res = clf.predict(t)
        print(f"'{t}'  →  {res['primary']} ({res['confidence']:.2f})")
