# 🏦 CreditSense — Prediksi Kelayakan Kredit Nasabah
**UAS Data Mining 2025 | CRISP-DM Framework**

## Struktur Folder

```
UAS_DataMining_NamaKelompok/
│
├── dataset/
│   ├── clean_credit_dataset.csv         ← dari notebook
│   └── clustered_credit_dataset.csv     ← dari notebook
│
├── model/
│   ├── random_forest.pkl                ← dari notebook
│   ├── kmeans.pkl                       ← dari notebook
│   ├── scaler.pkl                       ← dari notebook
│   └── encoders.pkl                     ← dari notebook
│
├── assets/                              ← gambar dari notebook (opsional)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── cluster_pca.png
│   ├── elbow_method.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_waterfall.png
│
├── notebook/
│   └── UAS_DataMining_Final_v2.ipynb
│
├── laporan/
│   └── laporan.pdf
│
├── app.py                               ← ⭐ Streamlit App
├── requirements.txt
└── README.md
```

## Cara Menjalankan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan aplikasi
streamlit run app.py
```

## Fitur Aplikasi

| Halaman | Konten |
|---------|--------|
| 🏠 Home | Judul proyek, deskripsi, identitas anggota, metode |
| 📊 Dataset Overview | Info dataset, statistik, visualisasi distribusi |
| 🔮 Prediction & Analysis | Form input nasabah → prediksi + cluster |
| 📈 Visualization | Evaluasi model, clustering, SHAP analysis |
| ℹ️ About | Penjelasan metode, dataset, informasi proyek |

## Catatan Penting

1. **Isi nama anggota** di `app.py` pada bagian Home dan About
2. **Letakkan file `.pkl`** dari Colab ke folder `model/`
3. **Letakkan file CSV** dari Colab ke folder `dataset/`
4. **Opsional:** Letakkan gambar hasil notebook ke folder `assets/` agar tampil di halaman Visualization

## Bonus Points yang Diraih

- ✅ **Explainable AI (SHAP)** — +2 poin
- ✅ **UI/UX Menarik** — +2 poin (dark theme, modern design)
- ✅ **Dashboard Interaktif** — +2 poin (form prediksi real-time)

---
*Dibuat dengan Streamlit · Python · Scikit-learn · SHAP*
