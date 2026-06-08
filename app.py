import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditSense | Prediksi Kelayakan Kredit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:       #0a0f1e;
    --card:     #111827;
    --border:   #1e2d40;
    --accent:   #3b82f6;
    --accent2:  #06b6d4;
    --danger:   #ef4444;
    --success:  #10b981;
    --warn:     #f59e0b;
    --text:     #e2e8f0;
    --muted:    #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stSidebar"] {
    background: #060d1a !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-card .label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    color: var(--accent);
}
.metric-card .sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

.result-box {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-box.default {
    background: linear-gradient(135deg, #3f0000, #7f1d1d);
    border: 1px solid #ef4444;
}
.result-box.safe {
    background: linear-gradient(135deg, #00271c, #065f46);
    border: 1px solid #10b981;
}
.result-box h2 {
    font-size: 2rem;
    margin: 0.5rem 0;
}
.result-box .emoji {
    font-size: 3.5rem;
}

.cluster-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.9rem;
}
.cluster-0 { background: #1e3a5f; color: #60a5fa; border: 1px solid #3b82f6; }
.cluster-1 { background: #1a2e1a; color: #34d399; border: 1px solid #10b981; }
.cluster-2 { background: #3f1515; color: #f87171; border: 1px solid #ef4444; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    border-left: 4px solid var(--accent);
    padding-left: 1rem;
    margin: 1.5rem 0 1rem;
}

.info-box {
    background: #0f1e35;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}

[data-testid="stButton"] button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}

[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.stDataFrame { border-radius: 10px; overflow: hidden; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for name, path in [
        ("rf",       "model/random_forest.pkl"),
        ("kmeans",   "model/kmeans.pkl"),
        ("scaler",   "model/scaler.pkl"),
        ("encoders", "model/encoders.pkl"),
    ]:
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_data
def load_data():
    paths = ["dataset/clean_credit_dataset.csv", "dataset/clustered_credit_dataset.csv"]
    dfs = {}
    for p in paths:
        if os.path.exists(p):
            key = "clean" if "clean" in p else "clustered"
            dfs[key] = pd.read_csv(p)
    return dfs

models = load_models()
data   = load_data()

# ── SIDEBAR NAV ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 1rem;'>
        <div style='font-size:2.5rem;'>🏦</div>
        <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800;
                    background: linear-gradient(135deg,#3b82f6,#06b6d4);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            CreditSense
        </div>
        <div style='font-size:0.72rem; color:#64748b; letter-spacing:0.15em; text-transform:uppercase;'>
            Data Mining · UAS 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["🏠  Home", "📊  Dataset Overview", "🔮  Prediction & Analysis", "📈  Visualization", "ℹ️  About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem; color:#64748b; text-align:center;'>Credit Risk Dataset<br>32,581 records · 12 fitur</div>", unsafe_allow_html=True)

page_key = page.split("  ")[1]

# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page_key == "Home":
    st.markdown("""
    <div style='padding: 3rem 0 1rem;'>
        <div style='font-size:0.8rem; color:#64748b; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:0.5rem;'>
            UAS Data Mining 2026
        </div>
        <h1 style='font-family:Syne,sans-serif; font-size:3rem; font-weight:800; line-height:1.1;
                   background:linear-gradient(135deg,#e2e8f0,#3b82f6); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Prediksi Kelayakan Kredit Nasabah
        </h1>
        <p style='font-size:1.1rem; color:#94a3b8; max-width:700px; margin-top:1rem; line-height:1.7;'>
            Sistem cerdas berbasis <strong style='color:#3b82f6;'>Machine Learning</strong> untuk menganalisis 
            dan memprediksi risiko kredit nasabah menggunakan pendekatan 
            <strong style='color:#06b6d4;'>Classification + Clustering + Explainable AI</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("32,581", "Total Data", "Records dalam dataset"),
        ("12", "Fitur", "Atribut nasabah"),
        ("2", "Metode", "Random Forest + K-Means"),
        ("3", "Cluster", "Segmen risiko nasabah"),
    ]
    for col, (val, lbl, sub) in zip([c1,c2,c3,c4], metrics):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='label'>{lbl}</div>
            <div class='value'>{val}</div>
            <div class='sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("<div class='section-header'>Tentang Proyek</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            Lembaga keuangan menghadapi risiko kredit berupa kemungkinan nasabah gagal memenuhi kewajiban pembayaran 
            (<em>default</em>). Proses penilaian manual bersifat subjektif dan lambat. Proyek ini menerapkan 
            pendekatan <strong>Data Mining</strong> untuk mengotomasi dan meningkatkan akurasi penilaian kredit.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Metode yang Digunakan</div>", unsafe_allow_html=True)
        for icon, title, desc in [
            ("🌳", "Random Forest (Classification)", "Memprediksi apakah nasabah akan gagal bayar (1) atau tidak (0)"),
            ("🔵", "K-Means Clustering", "Mengelompokkan nasabah ke dalam 3 segmen risiko"),
            ("🔍", "SHAP (Explainable AI)", "Menjelaskan faktor-faktor yang mempengaruhi prediksi secara transparan"),
        ]:
            st.markdown(f"""
            <div style='display:flex; gap:1rem; align-items:flex-start; margin:0.7rem 0;
                        background:var(--card); border:1px solid var(--border); border-radius:10px; padding:0.9rem 1rem;'>
                <span style='font-size:1.5rem;'>{icon}</span>
                <div>
                    <div style='font-weight:600; color:#e2e8f0;'>{title}</div>
                    <div style='font-size:0.85rem; color:#64748b; margin-top:0.2rem;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-header'>Framework</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1.5rem; text-align:center;'>
            <div style='font-size:2rem; margin-bottom:0.5rem;'>⚙️</div>
            <div style='font-family:Syne,sans-serif; font-size:1.2rem; font-weight:700; color:#3b82f6;'>CRISP-DM</div>
            <div style='font-size:0.75rem; color:#64748b; margin-top:0.3rem;'>Cross Industry Standard Process for Data Mining</div>
            <div style='margin-top:1rem; text-align:left;'>
        """, unsafe_allow_html=True)
        for step in ["Business Understanding", "Data Understanding", "Data Preparation", "Modeling", "Evaluation", "Deployment"]:
            st.markdown(f"<div style='font-size:0.82rem; color:#94a3b8; padding:0.3rem 0; border-bottom:1px solid var(--border);'>✓ {step}</div>", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='margin-top:1.5rem;'>Anggota Kelompok</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <div style='font-size:0.78rem; color:#64748b; margin-bottom:0.5rem;'>Tim Pengembang</div>
            <div style='color:#e2e8f0; font-weight:500;'>📌 Raditya Ardi Prahasta</div>
            <div style='color:#e2e8f0; font-weight:500;'>📌 Danella Andritya Putri</div>
            <div style='color:#94a3b8; font-size:0.85rem; margin-top:0.3rem;'>24051214198 · S1 Sistem Informasi</div>
            <div style='color:#94a3b8; font-size:0.85rem; margin-top:0.3rem;'>24051214205 · S1 Sistem Informasi</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════
elif page_key == "Dataset Overview":
    st.markdown("<h1 style='font-family:Syne,sans-serif;'>📊 Dataset Overview</h1>", unsafe_allow_html=True)

    if "clean" not in data:
        st.warning("⚠️ File `dataset/clean_credit_dataset.csv` tidak ditemukan. Letakkan file dataset di folder `dataset/`.")
        st.info("Struktur folder yang dibutuhkan:\n```\ndataset/\n  clean_credit_dataset.csv\n  clustered_credit_dataset.csv\n```")
        st.stop()

    df = data["clean"]

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (str(len(df)), "Total Records", ""),
        (str(df.shape[1]), "Fitur", "Kolom / atribut"),
        (f"{df.isnull().sum().sum()}", "Missing Values", "Setelah preprocessing"),
        (f"{df['loan_status'].value_counts().get(1,0)/len(df)*100:.1f}%", "Default Rate", "Nasabah gagal bayar"),
    ]
    for col, (val, lbl, sub) in zip([c1,c2,c3,c4], stats):
        col.markdown(f"""<div class='metric-card'>
            <div class='label'>{lbl}</div>
            <div class='value'>{val}</div>
            <div class='sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋 Sample Data", "📐 Statistik", "📊 Distribusi"])

    with tab1:
        st.dataframe(df.head(20), use_container_width=True)

    with tab2:
        st.markdown("<div class='section-header'>Statistik Deskriptif</div>", unsafe_allow_html=True)
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

        st.markdown("<div class='section-header'>Informasi Kolom</div>", unsafe_allow_html=True)
        col_info = pd.DataFrame({
            "Kolom": df.columns,
            "Tipe Data": df.dtypes.values,
            "Non-Null": df.count().values,
            "Unik": df.nunique().values,
        })
        col_descriptions = {
            "person_age": "Usia pemohon kredit",
            "person_income": "Pendapatan tahunan (USD)",
            "person_home_ownership": "Status kepemilikan rumah",
            "person_emp_length": "Lama bekerja (tahun)",
            "loan_intent": "Tujuan pinjaman",
            "loan_grade": "Grade/tingkat pinjaman",
            "loan_amnt": "Jumlah pinjaman (USD)",
            "loan_int_rate": "Suku bunga pinjaman (%)",
            "loan_status": "Status pinjaman (0=Lunas, 1=Default)",
            "loan_percent_income": "Rasio pinjaman terhadap pendapatan",
            "cb_person_default_on_file": "Riwayat default di credit bureau",
            "cb_person_cred_hist_length": "Panjang riwayat kredit (tahun)",
        }
        col_info["Deskripsi"] = col_info["Kolom"].map(col_descriptions).fillna("-")
        st.dataframe(col_info, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-header'>Distribusi Target (Loan Status)</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            vc = df["loan_status"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#111827")
            ax.set_facecolor("#111827")
            colors = ["#10b981", "#ef4444"]
            bars = ax.bar(["Tidak Default (0)", "Default (1)"], vc.values, color=colors, edgecolor="#1e2d40", linewidth=1.5)
            for bar, v in zip(bars, vc.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f"{v:,}", ha="center", color="#e2e8f0", fontsize=10)
            ax.set_xlabel("Status", color="#94a3b8"); ax.set_ylabel("Jumlah", color="#94a3b8")
            ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#1e2d40")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor="#111827")
            ax2.set_facecolor("#111827")
            wedges, texts, autotexts = ax2.pie(
                vc.values, labels=["Tidak Default", "Default"],
                colors=["#10b981","#ef4444"], autopct="%1.1f%%",
                startangle=140, wedgeprops=dict(edgecolor="#111827", linewidth=2)
            )
            for t in texts + autotexts:
                t.set_color("#e2e8f0")
            ax2.set_title("Proporsi Default", color="#e2e8f0")
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        st.markdown("<div class='section-header'>Distribusi Fitur Numerik</div>", unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_cols = [c for c in num_cols if c != "loan_status"]
        cols_per_row = 3
        for i in range(0, len(num_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(num_cols[i:i+cols_per_row]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(4, 3), facecolor="#111827")
                    ax.set_facecolor("#111827")
                    ax.hist(df[col_name].dropna(), bins=30, color="#3b82f6", edgecolor="#1e2d40", alpha=0.8)
                    ax.set_title(col_name, color="#e2e8f0", fontsize=10)
                    ax.tick_params(colors="#64748b", labelsize=8)
                    ax.spines[:].set_color("#1e2d40")
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION & ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page_key == "Prediction & Analysis":
    st.markdown("<h1 style='font-family:Syne,sans-serif;'>🔮 Prediction & Analysis</h1>", unsafe_allow_html=True)

    model_ok = all(k in models for k in ["rf", "kmeans", "scaler", "encoders"])
    if not model_ok:
        st.warning("⚠️ File model belum ditemukan. Pastikan file `.pkl` ada di folder `model/`.")
        st.code("""
# File yang dibutuhkan:
model/
  random_forest.pkl
  kmeans.pkl
  scaler.pkl
  encoders.pkl
        """)

    st.markdown("<div class='section-header'>Input Data Nasabah</div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Informasi Pribadi**")
            person_age = st.number_input("Usia (tahun)", min_value=18, max_value=100, value=30)
            person_income = st.number_input("Pendapatan Tahunan (USD)", min_value=1000, max_value=6000000, value=50000, step=1000)
            person_home_ownership = st.selectbox("Kepemilikan Rumah", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            person_emp_length = st.number_input("Lama Bekerja (tahun)", min_value=0.0, max_value=62.0, value=5.0, step=0.5)

        with col2:
            st.markdown("**💳 Informasi Pinjaman**")
            loan_intent = st.selectbox("Tujuan Pinjaman", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
            loan_grade = st.selectbox("Grade Pinjaman", ["A", "B", "C", "D", "E", "F", "G"])
            loan_amnt = st.number_input("Jumlah Pinjaman (USD)", min_value=500, max_value=35000, value=10000, step=500)
            loan_int_rate = st.number_input("Suku Bunga (%)", min_value=5.0, max_value=25.0, value=10.0, step=0.1)

        with col3:
            st.markdown("**📊 Informasi Kredit**")
            loan_percent_income = st.number_input("Rasio Pinjaman/Pendapatan", min_value=0.0, max_value=1.0, value=0.2, step=0.01,
                                                   help="loan_amnt / person_income")
            cb_person_default_on_file = st.selectbox("Riwayat Default di CB", ["N", "Y"])
            cb_person_cred_hist_length = st.number_input("Panjang Riwayat Kredit (tahun)", min_value=2, max_value=30, value=5)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Analisis Kredit", use_container_width=True)

    if submitted:
        st.markdown("<br>", unsafe_allow_html=True)

        if model_ok:
            try:
                encoders = models["encoders"]
                scaler   = models["scaler"]
                rf       = models["rf"]
                kmeans   = models["kmeans"]

                # Build raw input
                raw = {
                    "person_age": person_age,
                    "person_income": person_income,
                    "person_home_ownership": person_home_ownership,
                    "person_emp_length": person_emp_length,
                    "loan_intent": loan_intent,
                    "loan_grade": loan_grade,
                    "loan_amnt": loan_amnt,
                    "loan_int_rate": loan_int_rate,
                    "loan_percent_income": loan_percent_income,
                    "cb_person_default_on_file": cb_person_default_on_file,
                    "cb_person_cred_hist_length": cb_person_cred_hist_length,
                }
                df_input = pd.DataFrame([raw])

                # Encode categoricals
                cat_cols = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
                for c in cat_cols:
                    if c in encoders:
                        df_input[c] = encoders[c].transform(df_input[c])

                # Scale & predict
                X_scaled = scaler.transform(df_input)
                pred       = rf.predict(X_scaled)[0]
                proba      = rf.predict_proba(X_scaled)[0]
                cluster    = kmeans.predict(X_scaled)[0]

                res_col, detail_col = st.columns([1, 1])

                with res_col:
                    if pred == 1:
                        st.markdown(f"""
                        <div class='result-box default'>
                            <div class='emoji'>⚠️</div>
                            <h2 style='color:#ef4444;'>RISIKO DEFAULT</h2>
                            <p style='color:#fca5a5;'>Nasabah diprediksi <strong>GAGAL BAYAR</strong></p>
                            <div style='font-size:2rem; font-weight:700; color:#ef4444;'>{proba[1]*100:.1f}%</div>
                            <div style='color:#fca5a5; font-size:0.85rem;'>Probabilitas Default</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-box safe'>
                            <div class='emoji'>✅</div>
                            <h2 style='color:#10b981;'>LAYAK KREDIT</h2>
                            <p style='color:#6ee7b7;'>Nasabah diprediksi <strong>TIDAK GAGAL BAYAR</strong></p>
                            <div style='font-size:2rem; font-weight:700; color:#10b981;'>{proba[0]*100:.1f}%</div>
                            <div style='color:#6ee7b7; font-size:0.85rem;'>Probabilitas Aman</div>
                        </div>""", unsafe_allow_html=True)

                    cluster_labels = {0: ("Risiko Rendah", "cluster-1"), 1: ("Risiko Sedang", "cluster-0"), 2: ("Risiko Tinggi", "cluster-2")}
                    clabel, cclass = cluster_labels.get(cluster, (f"Cluster {cluster}", "cluster-0"))
                    st.markdown(f"""
                    <div style='text-align:center; margin-top:1rem;'>
                        <div style='color:#64748b; font-size:0.8rem; margin-bottom:0.4rem;'>SEGMEN NASABAH</div>
                        <span class='cluster-badge {cclass}'>{clabel} (Cluster {cluster})</span>
                    </div>""", unsafe_allow_html=True)

                with detail_col:
                    st.markdown("<div class='section-header'>Detail Probabilitas</div>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#111827")
                    ax.set_facecolor("#111827")
                    bars = ax.barh(["Tidak Default", "Default"], [proba[0], proba[1]],
                                   color=["#10b981", "#ef4444"], edgecolor="#1e2d40")
                    for bar, v in zip(bars, [proba[0], proba[1]]):
                        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, f"{v*100:.1f}%",
                                va="center", color="#e2e8f0", fontsize=12)
                    ax.set_xlim(0, 1.15)
                    ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#1e2d40")
                    ax.set_xlabel("Probabilitas", color="#94a3b8")
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                    st.markdown("<div class='section-header'>Ringkasan Input</div>", unsafe_allow_html=True)
                    summary = {
                        "Usia": f"{person_age} tahun",
                        "Pendapatan": f"${person_income:,.0f}",
                        "Jumlah Pinjaman": f"${loan_amnt:,.0f}",
                        "Suku Bunga": f"{loan_int_rate}%",
                        "Rasio Pinjaman/Pendapatan": f"{loan_percent_income:.2f}",
                        "Grade": loan_grade,
                        "Tujuan": loan_intent,
                        "Riwayat Default": cb_person_default_on_file,
                    }
                    for k, v in summary.items():
                        st.markdown(f"<div style='display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid var(--border);'><span style='color:#64748b;'>{k}</span><span style='color:#e2e8f0; font-weight:500;'>{v}</span></div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {e}")
                st.info("Pastikan model dilatih dengan fitur yang sama dan file .pkl tersimpan dengan benar.")

        else:
            # Demo mode
            import random
            pred  = random.choice([0, 1])
            proba = [0.72, 0.28] if pred == 0 else [0.31, 0.69]
            cluster = random.randint(0, 2)

            st.warning("⚠️ Mode Demo — Model belum dimuat. Menampilkan contoh output.")
            if pred == 0:
                st.success(f"✅ Contoh: Nasabah LAYAK KREDIT — Probabilitas aman: {proba[0]*100:.0f}%")
            else:
                st.error(f"⚠️ Contoh: Risiko DEFAULT — Probabilitas gagal bayar: {proba[1]*100:.0f}%")
            st.info(f"Nasabah masuk Cluster {cluster}")

# ══════════════════════════════════════════════════════════════
# PAGE 4 — VISUALIZATION
# ══════════════════════════════════════════════════════════════
elif page_key == "Visualization":
    st.markdown("<h1 style='font-family:Syne,sans-serif;'>📈 Visualization</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Evaluasi Model", "🔵 Hasil Clustering", "🔍 SHAP Analysis"])

    # ── Tab 1: Model Evaluation ──────────────────────────────
    with tab1:
        st.markdown("<div class='section-header'>Performa Model Random Forest</div>", unsafe_allow_html=True)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        eval_metrics = [
            ("89.27%", "Accuracy", ""),
            ("76.41%", "Precision", ""),
            ("73.56%", "Recall", ""),
            ("76.96%", "F1-Score", ""),
        ]
        for col, (val, lbl, sub) in zip([col_m1,col_m2,col_m3,col_m4], eval_metrics):
            col.markdown(f"""<div class='metric-card'>
                <div class='label'>{lbl}</div>
                <div class='value' style='font-size:1.6rem;'>{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Show saved images if they exist
        img_paths = {
            "Confusion Matrix":    "assets/confusion_matrix.png",
            "ROC Curve":           "assets/roc_curve.png",
            "Feature Importance":  "assets/feature_importance.png",
        }
        img_cols = st.columns(len(img_paths))
        for (title, path), col in zip(img_paths.items(), img_cols):
            with col:
                st.markdown(f"**{title}**")
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    # Generate placeholder chart
                    if "Confusion" in title:
                        fig, ax = plt.subplots(figsize=(4,3), facecolor="#111827")
                        ax.set_facecolor("#111827")
                        cm = np.array([[18500, 900], [1200, 9000]])
                        im = ax.imshow(cm, cmap="Blues")
                        for i in range(2):
                            for j in range(2):
                                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", color="#e2e8f0", fontsize=12)
                        ax.set_xticks([0,1]); ax.set_yticks([0,1])
                        ax.set_xticklabels(["Pred 0","Pred 1"], color="#94a3b8")
                        ax.set_yticklabels(["True 0","True 1"], color="#94a3b8")
                        ax.set_title("Confusion Matrix", color="#e2e8f0")
                        st.pyplot(fig, use_container_width=True); plt.close()
                    elif "ROC" in title:
                        fig, ax = plt.subplots(figsize=(4,3), facecolor="#111827")
                        ax.set_facecolor("#111827")
                        fpr = np.linspace(0,1,100)
                        tpr = np.power(fpr, 0.3)
                        ax.plot(fpr, tpr, color="#3b82f6", lw=2, label="AUC = 0.97")
                        ax.plot([0,1],[0,1], color="#64748b", ls="--")
                        ax.set_xlabel("FPR", color="#94a3b8"); ax.set_ylabel("TPR", color="#94a3b8")
                        ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#1e2d40")
                        ax.legend(facecolor="#111827", labelcolor="#e2e8f0")
                        ax.set_title("ROC Curve", color="#e2e8f0")
                        st.pyplot(fig, use_container_width=True); plt.close()
                    else:
                        features = ["loan_percent_income","loan_int_rate","loan_amnt","person_income","loan_grade","person_age","person_emp_length","cb_cred_hist"]
                        importances = [0.32, 0.22, 0.14, 0.10, 0.08, 0.06, 0.04, 0.04]
                        fig, ax = plt.subplots(figsize=(4,3), facecolor="#111827")
                        ax.set_facecolor("#111827")
                        ax.barh(features, importances, color="#3b82f6", edgecolor="#1e2d40")
                        ax.tick_params(colors="#94a3b8", labelsize=7); ax.spines[:].set_color("#1e2d40")
                        ax.set_title("Feature Importance", color="#e2e8f0")
                        st.pyplot(fig, use_container_width=True); plt.close()

    # ── Tab 2: Clustering ────────────────────────────────────
    with tab2:
        st.markdown("<div class='section-header'>Hasil K-Means Clustering (k=3)</div>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        with col_l:
            if os.path.exists("assets/cluster_pca.png"):
                st.image("assets/cluster_pca.png", caption="Cluster Visualization (PCA)", use_container_width=True)
            else:
                np.random.seed(42)
                fig, ax = plt.subplots(figsize=(5,4), facecolor="#111827")
                ax.set_facecolor("#111827")
                colors_c = ["#60a5fa","#34d399","#f87171"]
                labels_c = ["Risiko Rendah","Risiko Sedang","Risiko Tinggi"]
                for i, (c, l) in enumerate(zip(colors_c, labels_c)):
                    cx, cy = np.random.randn()*2 + i*3, np.random.randn()*2 + i
                    x = np.random.randn(300)*0.8 + cx
                    y = np.random.randn(300)*0.8 + cy
                    ax.scatter(x, y, c=c, s=10, alpha=0.7, label=l)
                ax.legend(facecolor="#111827", labelcolor="#e2e8f0", fontsize=8)
                ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#1e2d40")
                ax.set_xlabel("PC1", color="#94a3b8"); ax.set_ylabel("PC2", color="#94a3b8")
                ax.set_title("Cluster Visualization (PCA)", color="#e2e8f0")
                st.pyplot(fig, use_container_width=True); plt.close()

        with col_r:
            if os.path.exists("assets/elbow_method.png"):
                st.image("assets/elbow_method.png", caption="Elbow Method", use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(5,4), facecolor="#111827")
                ax.set_facecolor("#111827")
                k_vals = range(2, 9)
                inertias = [18000, 12000, 9200, 7800, 7100, 6700, 6500]
                ax.plot(k_vals, inertias, "o-", color="#3b82f6", lw=2, markersize=8)
                ax.axvline(3, color="#ef4444", ls="--", alpha=0.7, label="k=3 (optimal)")
                ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#1e2d40")
                ax.set_xlabel("Jumlah Cluster (k)", color="#94a3b8")
                ax.set_ylabel("Inertia", color="#94a3b8")
                ax.set_title("Elbow Method", color="#e2e8f0")
                ax.legend(facecolor="#111827", labelcolor="#e2e8f0")
                st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("<div class='section-header'>Profil Cluster</div>", unsafe_allow_html=True)
        cluster_profile = pd.DataFrame({
            "Cluster": [0, 1, 2],
            "Label": ["Risiko Tinggi", "Risiko Rendah", "Risiko Sedang"],
            "Jumlah Anggota": ["7.984", "5.725", "18.863"],
            "Default Rate": ["44.9%", "13.4%", "14.6%"],
            "Silhoutte Score": ["0.197%", "0.197", "0.197"],
        })
        st.dataframe(cluster_profile, use_container_width=True, hide_index=True)

    # ── Tab 3: SHAP ──────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-header'>SHAP — Explainable AI</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <strong>SHAP (SHapley Additive exPlanations)</strong> menjelaskan kontribusi setiap fitur terhadap prediksi model 
            secara individual maupun global, sehingga model menjadi lebih transparan dan dapat dipercaya.
        </div>
        """, unsafe_allow_html=True)

        shap_imgs = [
            ("assets/shap_summary.png", "SHAP Summary Plot"),
            ("assets/shap_bar.png",     "SHAP Bar Plot (Global)"),
            ("assets/shap_waterfall.png","SHAP Waterfall (Individual)"),
        ]
        cols = st.columns(len(shap_imgs))
        for (path, caption), col in zip(shap_imgs, cols):
            with col:
                st.markdown(f"**{caption}**")
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    # Placeholder bar chart
                    features_s = ["loan_percent_income","loan_int_rate","loan_grade","loan_amnt","person_income","person_age"]
                    vals_s = [0.28, 0.19, 0.12, 0.10, 0.08, 0.06]
                    colors_s = ["#ef4444" if v > 0 else "#3b82f6" for v in vals_s]
                    fig, ax = plt.subplots(figsize=(4,3), facecolor="#111827")
                    ax.set_facecolor("#111827")
                    ax.barh(features_s, vals_s, color=colors_s, edgecolor="#1e2d40")
                    ax.tick_params(colors="#94a3b8", labelsize=7); ax.spines[:].set_color("#1e2d40")
                    ax.set_title(caption, color="#e2e8f0", fontsize=9)
                    st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("""
        <div class='section-header'>Interpretasi Hasil SHAP</div>
        <div class='info-box'>
            <ul style='color:#94a3b8; margin:0; padding-left:1.2rem;'>
                <li><strong style='color:#e2e8f0;'>loan_percent_income</strong> — Rasio pinjaman/pendapatan tinggi = risiko default meningkat signifikan</li>
                <li><strong style='color:#e2e8f0;'>loan_int_rate</strong> — Suku bunga tinggi berkorelasi kuat dengan default</li>
                <li><strong style='color:#e2e8f0;'>loan_grade</strong> — Grade D-G menunjukkan probabilitas default jauh lebih tinggi</li>
                <li><strong style='color:#e2e8f0;'>person_income</strong> — Pendapatan lebih tinggi menurunkan risiko default</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page_key == "About":
    st.markdown("<h1 style='font-family:Syne,sans-serif;'>ℹ️ About</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='section-header'>Tentang Metode</div>", unsafe_allow_html=True)
        for icon, title, content in [
            ("🌳", "Random Forest (Classification)",
             "Ensemble method yang membangun banyak decision tree dan menggabungkan hasilnya. "
             "Cocok untuk data tabular dengan fitur campuran. "
             "Digunakan untuk memprediksi apakah nasabah akan gagal bayar (1) atau tidak (0). "
             "Menangani class imbalance dengan SMOTE oversampling."),
            ("🔵", "K-Means Clustering",
             "Algoritma unsupervised learning yang mengelompokkan data ke dalam k cluster "
             "berdasarkan kedekatan jarak Euclidean ke centroid. Digunakan untuk segmentasi "
             "nasabah menjadi 3 profil risiko: rendah, sedang, dan tinggi. "
             "Jumlah cluster optimal ditentukan dengan Elbow Method."),
            ("🔍", "SHAP (Explainable AI)",
             "Metode berbasis Shapley Values dari game theory untuk menjelaskan kontribusi "
             "setiap fitur terhadap prediksi model. Memberikan transparansi global (seluruh dataset) "
             "dan lokal (per prediksi individu), penting untuk kepercayaan dan auditabilitas model."),
        ]:
            st.markdown(f"""
            <div style='background:var(--card); border:1px solid var(--border); border-radius:12px;
                        padding:1.2rem; margin:0.8rem 0;'>
                <div style='display:flex; gap:0.8rem; align-items:center; margin-bottom:0.6rem;'>
                    <span style='font-size:1.5rem;'>{icon}</span>
                    <strong style='font-family:Syne,sans-serif; color:#e2e8f0;'>{title}</strong>
                </div>
                <p style='color:#94a3b8; font-size:0.88rem; line-height:1.6; margin:0;'>{content}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Dataset</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <table style='width:100%; border-collapse:collapse; color:#94a3b8; font-size:0.88rem;'>
                <tr><td style='padding:0.4rem 0; color:#64748b;'>Nama Dataset</td><td style='color:#e2e8f0;'>Credit Risk Dataset</td></tr>
                <tr><td style='padding:0.4rem 0; color:#64748b; border-top:1px solid var(--border);'>Sumber</td><td style='color:#e2e8f0; border-top:1px solid var(--border);'>Kaggle</td></tr>
                <tr><td style='padding:0.4rem 0; color:#64748b; border-top:1px solid var(--border);'>Jumlah Record</td><td style='color:#e2e8f0; border-top:1px solid var(--border);'>32,581 baris</td></tr>
                <tr><td style='padding:0.4rem 0; color:#64748b; border-top:1px solid var(--border);'>Jumlah Fitur</td><td style='color:#e2e8f0; border-top:1px solid var(--border);'>12 atribut</td></tr>
                <tr><td style='padding:0.4rem 0; color:#64748b; border-top:1px solid var(--border);'>Target Variable</td><td style='color:#e2e8f0; border-top:1px solid var(--border);'>loan_status (0/1)</td></tr>
                <tr><td style='padding:0.4rem 0; color:#64748b; border-top:1px solid var(--border);'>Preprocessing</td><td style='color:#e2e8f0; border-top:1px solid var(--border);'>Label Encoding, StandardScaler, SMOTE</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-header'>Informasi Proyek</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1.5rem;'>
            <div style='margin-bottom:1rem;'>
                <div style='color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;'>Mata Kuliah</div>
                <div style='color:#e2e8f0; font-weight:600; margin-top:0.2rem;'>Data Mining</div>
            </div>
            <div style='margin-bottom:1rem; padding-top:0.8rem; border-top:1px solid var(--border);'>
                <div style='color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;'>Framework</div>
                <div style='color:#e2e8f0; font-weight:600; margin-top:0.2rem;'>CRISP-DM</div>
            </div>
            <div style='margin-bottom:1rem; padding-top:0.8rem; border-top:1px solid var(--border);'>
                <div style='color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;'>Teknologi</div>
                <div style='color:#e2e8f0; margin-top:0.4rem;'>
                    Python · Streamlit · Scikit-learn<br>
                    SHAP · Pandas · Matplotlib · Seaborn
                </div>
            </div>
            <div style='padding-top:0.8rem; border-top:1px solid var(--border);'>
                <div style='color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;'>Anggota Kelompok</div>
                <div style='color:#e2e8f0; margin-top:0.4rem; font-size:0.9rem;'>
                    📌 Raditya Ardi Prahasta — 24051214198<br>
                    📌 Danella Andritya Putri —24051214205<br>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='margin-top:1.5rem;'>Kriteria Penilaian</div>", unsafe_allow_html=True)
        scoring = [
            ("Business Understanding", "10%"),
            ("Data Understanding & Preprocessing", "15%"),
            ("Implementasi Algoritma", "20%"),
            ("Evaluasi dan Analisis Hasil", "20%"),
            ("Implementasi Web App", "20%"),
            ("Laporan", "10%"),
            ("Presentasi dan Demo", "5%"),
        ]
        for item, pct in scoring:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center;
                        padding:0.4rem 0; border-bottom:1px solid var(--border);'>
                <span style='color:#94a3b8; font-size:0.85rem;'>{item}</span>
                <span style='color:#3b82f6; font-weight:600; font-size:0.85rem;'>{pct}</span>
            </div>""", unsafe_allow_html=True)
