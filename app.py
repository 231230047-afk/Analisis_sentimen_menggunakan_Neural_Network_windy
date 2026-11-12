"""
Aplikasi Analisis Sentimen Bahasa Indonesia
Neural Network (MLP) + TF-IDF + Cleansing + Normalisasi

Dataset:
data_tugas_sentimen.csv

Jalankan:
streamlit run app.py
"""

import re
import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="💬",
    layout="wide",
)

DATA_FILE = "data_tugas_sentimen.csv"

EMOJI = {
    "positif": "😊",
    "netral": "😐",
    "negatif": "😠",
}

# ============================================================
# NORMALISASI KATA
# ============================================================

NORMALISASI = {
    "yg": "yang",
    "yng": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "dr": "dari",
    "utk": "untuk",
    "untk": "untuk",
    "krn": "karena",
    "karna": "karena",
    "krena": "karena",
    "tdk": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "bgt": "banget",
    "banget": "banget",
    "klo": "kalau",
    "kalo": "kalau",
    "kl": "kalau",
    "aja": "saja",
    "jgn": "jangan",
    "jgn2": "jangan",
    "sm": "sama",
    "sama2": "sama-sama",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "blm": "belum",
    "belom": "belum",
    "dpt": "dapat",
    "dapet": "dapat",
    "jd": "jadi",
    "jadi2": "jadi",
    "org": "orang",
    "orng": "orang",
    "bbrp": "beberapa",
    "byk": "banyak",
    "bnyk": "banyak",
    "mksh": "terima kasih",
    "makasih": "terima kasih",
    "trims": "terima kasih",
    "thx": "terima kasih",
    "pls": "tolong",
    "please": "tolong",
    "info": "informasi",
    "yg2": "yang",
}

# ============================================================
# PREPROCESSING
# ============================================================

def bersihkan_teks(teks):
    """Cleansing sesuai alur notebook: URL, mention, hashtag,
    angka, karakter non-huruf, spasi berlebih, dan lowercase."""
    teks = str(teks)
    teks = re.sub(r"\[.*?\]", " ", teks)
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks)
    teks = re.sub(r"@\w+", " ", teks)
    teks = re.sub(r"#\w+", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)
    teks = teks.lower()
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


def normalisasi_teks(teks):
    """Mengubah kata singkatan/tidak baku menjadi bentuk baku."""
    tokens = teks.split()
    hasil = []

    for token in tokens:
        # Hilangkan pengulangan huruf berlebihan:
        # baguuuus -> bagus, baguuusss -> bagus
        token = re.sub(r"(.)\1{2,}", r"\1", token)

        # Normalisasi berdasarkan kamus
        token = NORMALISASI.get(token, token)
        hasil.append(token)

    return " ".join(hasil)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    if "full_text" not in df.columns:
        raise ValueError("Kolom 'full_text' tidak ditemukan.")

    if "sentimen" not in df.columns:
        raise ValueError("Kolom 'sentimen' tidak ditemukan.")

    df["full_text"] = df["full_text"].fillna("").astype(str)
    df["sentimen"] = df["sentimen"].fillna("").astype(str).str.lower().str.strip()

    # Cleansing dan normalisasi
    df["clean_text"] = df["full_text"].apply(bersihkan_teks)
    df["normalisasi_text"] = df["clean_text"].apply(normalisasi_teks)

    # Buang teks kosong
    df = df[df["normalisasi_text"].str.strip() != ""].copy()

    return df.reset_index(drop=True)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(df):
    X = df["normalisasi_text"]
    y = df["sentimen"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # TF-IDF dengan normalisasi L2
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Neural Network / MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    mlp.fit(X_train_tfidf, y_train)

    y_pred = mlp.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": labels,
        "cm": cm,
        "y_test": y_test,
        "y_pred": y_pred,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "vocab_size": len(tfidf.vocabulary_),
    }

    return tfidf, mlp, metrics


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Informasi Model")
st.sidebar.write("**Metode:** Neural Network / MLP")
st.sidebar.write("**Ekstraksi fitur:** TF-IDF")
st.sidebar.write("**Normalisasi:** L2 + normalisasi kata")
st.sidebar.write("**Dataset:** data_tugas_sentimen.csv")

# ============================================================
# HEADER
# ============================================================

st.title("💬 Analisis Sentimen")
st.caption("Neural Network (MLP) + TF-IDF + Cleansing + Normalisasi")

try:
    df = load_data()
    tfidf, mlp, metrics = train_model(df)
except FileNotFoundError:
    st.error(
        "File data_tugas_sentimen.csv tidak ditemukan. "
        "Letakkan file CSV di folder yang sama dengan app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
    st.stop()

# ============================================================
# RINGKASAN DATA
# ============================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah Data", f"{len(df):,}")
c2.metric("Data Training", f"{metrics['train_size']:,}")
c3.metric("Data Testing", f"{metrics['test_size']:,}")
c4.metric("Akurasi", f"{metrics['accuracy']:.2%}")

st.divider()

# ============================================================
# TAB
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Data",
        "🧹 Cleansing & Normalisasi",
        "🤖 Prediksi",
        "📈 Evaluasi Model",
    ]
)

# ============================================================
# TAB 1 - DATA
# ============================================================

with tab1:
    st.subheader("Dataset")

    st.write(
        f"Dataset berisi **{len(df):,} data** setelah proses preprocessing."
    )

    tampil = df.copy()

    st.dataframe(
        tampil[
            [
                c for c in
                ["created_at", "username", "full_text", "sentimen"]
                if c in tampil.columns
            ]
        ].head(100),
        use_container_width=True,
        height=450,
    )

    st.subheader("Distribusi Sentimen")
    distribusi = df["sentimen"].value_counts()
    st.bar_chart(distribusi)

# ============================================================
# TAB 2 - CLEANSING & NORMALISASI
# ============================================================

with tab2:
    st.subheader("Hasil Cleansing dan Normalisasi")

    st.info(
        "Kolom **normalisasi_text** adalah teks yang sudah melalui "
        "cleansing dan normalisasi kata."
    )

    kolom_tampil = [
        c for c in
        ["full_text", "clean_text", "normalisasi_text", "sentimen"]
        if c in df.columns
    ]

    st.dataframe(
        df[kolom_tampil].head(100),
        use_container_width=True,
        height=500,
    )

    # Download hasil preprocessing
    csv_normalisasi = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Data Cleansing + Normalisasi",
        data=csv_normalisasi,
        file_name="data_tugas_sentimen_normalisasi.csv",
        mime="text/csv",
    )

    st.subheader("Contoh Kamus Normalisasi")

    kamus_df = pd.DataFrame(
        list(NORMALISASI.items()),
        columns=["kata_awal", "kata_normalisasi"],
    )

    st.dataframe(
        kamus_df,
        use_container_width=True,
        height=300,
    )

# ============================================================
# TAB 3 - PREDIKSI
# ============================================================

with tab3:
    st.subheader("Prediksi Sentimen")

    teks = st.text_area(
        "Masukkan teks:",
        placeholder=(
            "Contoh: Kondisi udara hari ini bagus banget, "
            "semoga tetap seperti ini..."
        ),
        height=150,
    )

    if st.button("🔍 Analisis Sentimen", type="primary"):
        if not teks.strip():
            st.warning("Silakan masukkan teks terlebih dahulu.")
        else:
            clean = bersihkan_teks(teks)
            normal = normalisasi_teks(clean)

            if not normal:
                st.warning("Teks menjadi kosong setelah preprocessing.")
            else:
                X_input = tfidf.transform([normal])

                hasil = mlp.predict(X_input)[0]
                proba = mlp.predict_proba(X_input)[0]

                st.subheader("Hasil Preprocessing")

                p1, p2 = st.columns(2)

                with p1:
                    st.markdown("**Teks asli**")
                    st.write(teks)

                with p2:
                    st.markdown("**Setelah cleansing**")
                    st.write(clean)

                st.markdown("**Setelah normalisasi**")
                st.info(normal)

                st.divider()

                emoji = EMOJI.get(hasil, "💬")

                st.success(
                    f"### {emoji} {hasil.upper()}"
                )

                st.subheader("Probabilitas Sentimen")

                hasil_proba = pd.DataFrame(
                    {
                        "Sentimen": [
                            f"{EMOJI.get(c, '')} {c}"
                            for c in mlp.classes_
                        ],
                        "Probabilitas": proba,
                    }
                ).sort_values(
                    "Probabilitas",
                    ascending=False,
                )

                st.dataframe(
                    hasil_proba.style.format(
                        {"Probabilitas": "{:.2%}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.bar_chart(
                    hasil_proba.set_index("Sentimen")[
                        "Probabilitas"
                    ]
                )

# ============================================================
# TAB 4 - EVALUASI
# ============================================================

with tab4:
    st.subheader("Evaluasi Model Neural Network")

    e1, e2, e3, e4 = st.columns(4)

    e1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    e2.metric("Precision", f"{metrics['precision']:.2%}")
    e3.metric("Recall", f"{metrics['recall']:.2%}")
    e4.metric("F1-Score", f"{metrics['f1']:.2%}")

    st.write(
        f"Ukuran vocabulary TF-IDF: **{metrics['vocab_size']:,} fitur**"
    )

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        metrics["cm"],
        index=metrics["labels"],
        columns=metrics["labels"],
    )

    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Keterangan")
    st.write(
        "Baris menunjukkan kelas aktual, sedangkan kolom menunjukkan "
        "kelas hasil prediksi."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Analisis Sentimen menggunakan Neural Network (MLP) dengan "
    "TF-IDF, cleansing, dan normalisasi teks."
)
