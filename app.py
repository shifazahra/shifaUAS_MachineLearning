import streamlit as st
import numpy as np
import pandas as pd

# 1. Konfigurasi halaman
st.set_page_config(page_title="Shifa - Dashboard", page_icon="🌿", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #F4F7F6;
    }

    .st-emotion-cache-1r6slb0, .element-container {
        border-radius: 15px;
    }

    .main-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.02);
        margin-bottom: 20px;
        border: 1px solid #f0f0f0;
    }

    .metric-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        border: 1px solid #f0f0f0;
        text-align: left;
    }

    .highlight-green {
        color: #27AE60;
        font-weight: 700;
        font-size: 24px;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eee;
    }

    .stSlider [data-baseweb="slider"] {
        margin-bottom: 20px;
    }

    .report-step {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.01);
        margin-bottom: 25px;
        border: 1px solid #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)

# Logika prediksi
def predict_zona(n, p, k, t, h, ph, r):
    centers = {
        0: [100, 70, 45, 25, 80, 6.5, 200],
        1: [40, 40, 20, 28, 50, 6.0, 100],
        2: [20, 20, 15, 20, 20, 5.5, 50],
        3: [80, 50, 80, 25, 90, 7.0, 250],
        4: [60, 60, 40, 30, 75, 6.5, 150]
    }
    user_input = np.array([n, p, k, t, h, ph, r])
    distances = {z: np.linalg.norm(user_input - np.array(c)) for z, c in centers.items()}
    return min(distances, key=distances.get)

# 2. Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #2C3E50;'>🌿 ShifaUAS </h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("🔍 **Parameter Input Lahan**")

    in_n = st.slider("Nitrogen (N)", 0, 150, 50)
    in_p = st.slider("Phosphorus (P)", 0, 150, 50)
    in_k = st.slider("Potassium (K)", 0, 250, 50)
    in_temp = st.slider("Suhu (°C)", 10.0, 45.0, 25.0)
    in_hum = st.slider("Kelembaban (%)", 10.0, 100.0, 70.0)
    in_ph = st.slider("pH Tanah", 3.5, 10.0, 6.5)
    in_rain = st.slider("Curah Hujan (mm)", 20.0, 300.0, 100.0)

# 3. Header utama
st.markdown("<h1 style='color: #2C3E50;'>Precision Agriculture Analytics</h1>", unsafe_allow_html=True)
st.markdown("Real-time cluster mapping for soil and climate data.")

# Prediksi output
predicted_cluster = predict_zona(in_n, in_p, in_k, in_temp, in_hum, in_ph, in_rain)

st.markdown(f"""
    <div class="main-card" style="border-left: 8px solid #27AE60;">
        <p style="margin:0; color: #7F8C8D; font-weight: 500;">HASIL ANALISIS REKOMENDASI LAHAN BARU</p>
        <h2 style="margin:0; color: #2C3E50;">Zona Pertanian Ideal: <span style="color: #27AE60;">Zona {predicted_cluster}</span></h2>
        <p style="margin-top:10px; color: #5D6D7E;">Sistem mendeteksi kombinasi hara dan cuaca yang Anda masukkan paling identik dengan karakteristik <b>Zona {predicted_cluster}</b>.</p>
    </div>
    """, unsafe_allow_html=True)

# 4. Row
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.markdown('<div class="metric-box"><p style="color:#7F8C8D; font-size:12px;">TOTAL DATA</p><p class="highlight-green">2,200 Data</p><p style="color:#27AE60; font-size:11px;">22 Jenis Tanaman</p></div>', unsafe_allow_html=True)
with m_col2:
    st.markdown('<div class="metric-box"><p style="color:#7F8C8D; font-size:12px;">OPTIMAL K</p><p class="highlight-green">5 Zona</p><p style="color:#27AE60; font-size:11px;">Algoritma K-Means</p></div>', unsafe_allow_html=True)
with m_col3:
    st.markdown('<div class="metric-box"><p style="color:#7F8C8D; font-size:12px;">K-MEANS SILHOUETTE</p><p class="highlight-green">0.4762</p><p style="color:#27AE60; font-size:11px;">Model Alokasi Bisnis</p></div>', unsafe_allow_html=True)
with m_col4:
    st.markdown('<div class="metric-box"><p style="color:#7F8C8D; font-size:12px;">DBSCAN SILHOUETTE</p><p class="highlight-green">0.4771</p><p style="color:#27AE60; font-size:11px;">Model Deteksi Anomali</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 5. Struktur tabs visualisasi
tab1, tab2, tab3 = st.tabs(["📊 Data Science Report Pipeline", "🕸️ Cluster Profiling Analysis", "📋 Action Plan & Recommendation"])

# TAB 1
with tab1:
    st.markdown("<p style='color:#7F8C8D; font-size:14px; font-weight:bold;'>PIPELINE PERKEMBANGAN MODEL DARI DATA MENTAH HINGGA EVALUASI METRIK</p>", unsafe_allow_html=True)

    # Tahap 1: Sebaran Data Awal
    st.markdown("""
    <div class="report-step">
        <h4 style="color:#2C3E50; margin-bottom:5px;">Step 1: Eksplorasi Sebaran Data Asli (EDA)</h4>
        <p style="color:#7F8C8D; font-size:13px; margin-bottom:15px;">Melihat pola sebaran awal data hara (N, P, K) dan kondisi iklim makro sebelum dilakukan rekayasa fitur.</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        st.image("distribusi_fitur.png", use_column_width=True)
    except:
        st.info("Hubungkan file gambar distribusi awal Anda dengan nama 'distribusi_fitur.png'")

    # Tahap 2: Feature Engineering
    st.markdown("""
    <div class="report-step">
        <h4 style="color:#2C3E50; margin-bottom:5px;">Step 2: Rekayasa Fitur (Feature Engineering)</h4>
        <p style="color:#7F8C8D; font-size:13px; margin-bottom:15px;">Pembentukan fitur baru <code>NP_Ratio</code> dan <code>Water_Retention_Index</code> guna memperkuat pemisahan pola kluster.</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        st.image("feature_engineering.png", use_column_width=True)
    except:
        st.info("Hubungkan file gambar korelasi fitur Anda dengan nama 'feature_engineering.png'")

    # Tahap 3: PCA Dimensi Reduksi
    st.markdown("""
    <div class="report-step">
        <h4 style="color:#2C3E50; margin-bottom:5px;">Step 3: Reduksi Dimensi Menggunakan PCA</h4>
        <p style="color:#7F8C8D; font-size:13px; margin-bottom:15px;">Mentransformasikan seluruh fitur ke dalam koordinat PC1 dan PC2 guna mengatasi korelasi antar-variabel.</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        st.image("pca_biplot.png", use_column_width=True)
    except:
        st.info("Hubungkan file gambar biplot PCA Anda dengan nama 'pca_biplot.png'")

    # Tahap 4: Evaluasi Perbandingan Algoritma
    st.markdown("""
    <div class="report-step">
        <h4 style="color:#2C3E50; margin-bottom:5px;">Step 4: Evaluasi Perbandingan Metode Clustering</h4>
        <p style="color:#7F8C8D; font-size:13px; margin-bottom:15px;">Perbandingan jumlah pembentukan kluster optimal antara K-Means, Hierarchical, dan DBSCAN.</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        st.image("evaluasi_model.png", use_column_width=True)
    except:
        st.info("Hubungkan file bar chart perbandingan kluster Anda dengan nama 'evaluasi_model.png'")

# TAB 2
with tab2:
    st.markdown("<p style='color:#7F8C8D; font-size:14px; font-weight:bold;'>ANALISIS SEBARAN KARAKTERISTIK UNTUK TIAP MODEL</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class="report-step">
        <h4 style="color:#2C3E50; margin-bottom:5px;">1. Spider Plot / Radar Chart Profiling</h4>
        <p style="color:#7F8C8D; font-size:13px; margin-bottom:15px;">Melihat tarikan jaring parameter hara dan cuaca dominan pada masing-masing kluster model K-Means vs DBSCAN.</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        st.image("radar_km_dbscan.png", use_column_width=True)
    except:
        st.info("Hubungkan file radar chart Anda dengan nama 'radar_km_dbscan.png'")

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="report-step">
        <h4 style="color:#2C3E50; margin-bottom:5px;">2. Heatmap Kepadatan Nilai Fitur (Normalized)</h4>
        <p style="color:#7F8C8D; font-size:13px; margin-bottom:15px;">Matriks gradasi warna nilai rata-rata tiap komponen lahan untuk memvalidasi perbedaan batas zona komoditas tanaman.</p>
    </div>
    """, unsafe_allow_html=True)
    try:
        st.image("cluster_heatmaps_comparison.png", use_column_width=True)
    except:
        st.info("Hubungkan file heatmap komparatif Anda dengan nama 'cluster_heatmaps_comparison.png'")

# TAB 3
with tab3:
    st.markdown(f"### 📋 Panduan Aksi & Strategi Bisnis - Zona {predicted_cluster}")

    if predicted_cluster == 0:
        st.error("📍 ZONA 0 - KELOMPOK HORTIKULTURA DATARAN TINGGI (P & K Super Tinggi)")
        st.markdown("""
        * **Komoditas Paling Cocok:** Grapes (Anggur), Apple (Apel).
        * **Karakteristik Lahan:** Kandungan Fosfor dan Kalium alami sangat dominan, sangat memicu percepatan fase pembuahan tanaman batang kayu.
        * **Rekomendasi Aksi Bisnis:**
            * ✅ Struktur hara sudah sangat subur, kurangi pupuk kimia buatan dan optimalkan pemberian pupuk organik cair.
            * 🌿 Wajib menjaga sistem drainase dan kebersihan parit lahan agar akar buah tidak membusuk akibat pengendapan air berlebih.
        """)
    elif predicted_cluster == 1:
        st.warning("📍 ZONA 1 - KELOMPOK BUAH SEMUSIM (Iklim Hangat & Air Sedang)")
        st.markdown("""
        * **Komoditas Paling Cocok:** Watermelon (Semangka), Muskmelon (Melon), Cotton (Kapas).
        * **Karakteristik Lahan:** Nilai suhu lingkungan cenderung hangat konstan, sangat membantu proses pembentukan kadar fruktosa (gula) alami buah.
        * **Rekomendasi Aksi Bisnis:**
            * 💧 Lakukan sistem irigasi tetes secara berkala guna mengantisipasi fluktuasi kelembaban tanah yang ekstrem.
        """)
    elif predicted_cluster == 2:
        st.success("📍 ZONA 2 - ZONA KONSERVASI NITROGEN ALAMI (Kelompok Kacang-Kacangan)")
        st.markdown("""
        * **Komoditas Paling Cocok:** Chickpea, Lentil, Kidneybeans (Kacang Merah).
        * **Karakteristik Lahan:** Tingkat kelembaban cenderung rendah namun kaya pasokan nitrogen alami berkat kemampuan simbiosis akar legum dengan bakteri tanah.
        * **Rekomendasi Aksi Bisnis:**
            * ✅ Sangat direkomendasikan menjadi area rotasi tanaman berkala guna memulihkan tingkat kesuburan senyawa makro tanah tanpa ketergantungan pupuk urea.
        """)
    elif predicted_cluster == 3:
        st.info("📍 ZONA 3 - ZONA LAHAN BASAH & SAWAH INUNDASI (Curah Hujan Tinggi)")
        st.markdown("""
        * **Komoditas Paling Cocok:** Rice (Padi), Jute (Serat Goni), Coconut.
        * **Karakteristik Lahan:** Nilai curah hujan harian tinggi serta retensi indeks penampungan air maksimal.
        * **Rekomendasi Aksi Bisnis:**
            * 🌿 Maksimalkan manajemen sistem pintu tata air sawah berkala untuk mencegah pengendapan asam berlebih pada akar tanaman padi.
        """)
    elif predicted_cluster == 4:
        st.markdown("<div style='background-color: #FCF3CF; padding: 15px; border-radius: 5px; border-left: 5px solid #F1C40F;'><strong>📍 ZONA 4 - KELOMPOK PERKEBUNAN TROPIS (Suhu Stabil & Hara Netral)</strong></div>", unsafe_allow_html=True)
        st.markdown("""
        * **Komoditas Paling Cocok:** Banana (Pisang), Orange (Jeruk), Pomegranate (Delima).
        * **Karakteristik Lahan:** Berada pada cakupan pH netral ideal dengan kondisi suhu hangat stabil sepanjang tahun.
        * **Rekomendasi Aksi Bisnis:**
            * ✅ Kondisi ideal untuk investasi perkebunan jangka panjang. Lakukan pemupukan hara mikro NPK secara berkala pada fase awal tanam guna memperkuat pondasi batang.
        """)

# FOOTER CREDITS
st.markdown("---")
st.markdown("<p style='text-align: center; color: #BDC3C7;'>© 2026 PinenAgro Precision Agriculture System — UNSIKA Project</p>", unsafe_allow_html=True)
