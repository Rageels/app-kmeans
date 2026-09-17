import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Sistem Clustering K-Means",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# INJECT CUSTOM CSS (Repair tema sama teks invisible)
# =========================================================

def apply_custom_css():
    st.markdown("""
    <style>
    /* =========================================================
       BACKGROUND & TEMA UTAMA
    ========================================================= */
    .stApp {
        background-color: #F4F7FC !important;
    }

    /* Paksa semua teks di area utama menjadi gelap & jelas */
    .main p, .main span, .main label, 
    [data-testid="stMarkdownContainer"] p, 
    [data-testid="stMarkdownContainer"] span {
        color: #1D3557 !important;
    }

    /* Judul & Subjudul Utama */
    h1, h1 * {
        color: #0F4C81 !important;
        font-weight: 800 !important;
    }

    h2, h2 *, h3, h3 * {
        color: #1D3557 !important;
        font-weight: 700 !important;
    }

    /* =========================================================
       METRIC CARD (PERBAIKAN TEKS TRANSPARAN/PUTIH)
    ========================================================= */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        padding: 18px 20px !important;
        border-radius: 15px !important;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.08) !important;
        border-left: 6px solid #0F4C81 !important;
    }

    /* Label Judul Metric (misal: Total Mahasiswa) */
    div[data-testid="stMetricLabel"] *, 
    div[data-testid="stMetricLabel"] p {
        color: #555555 !important;
        font-size: 14px !important;
        font-weight: 700 !important;
    }

    /* Angka / Nilai Metric */
    div[data-testid="stMetricValue"] *, 
    div[data-testid="stMetricValue"] div {
        color: #0F4C81 !important;
        font-size: 26px !important;
        font-weight: 800 !important;
    }

    /* =========================================================
       SIDEBAR & RADIO MENU (PERBAIKAN MENU TERPOTONG)
    ========================================================= */
    section[data-testid="stSidebar"] {
        background-color: #0F4C81 !important;
    }

    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] caption {
        color: #FFFFFF !important;
    }

    /* Container Radio Menu */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        background-color: transparent !important;
        padding: 0px !important;
        border-radius: 0px !important;
        width: 100% !important;
    }

    /* Item Pilihan Menu Sidebar */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: rgba(255, 255, 255, 0.08) !important;
        margin-bottom: 6px !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        display: flex !important;
        align-items: center !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.22) !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label span {
        color: #FFFFFF !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }

    /* Tombol Logout Sidebar */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #D32F2F !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #B71C1C !important;
    }

    /* =========================================================
       FORM LOGIN CARD
    ========================================================= */
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        padding: 30px !important;
        border-radius: 20px !important;
        box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid #E2E8F0 !important;
    }

    div[data-testid="stForm"] label, 
    div[data-testid="stForm"] p {
        color: #1D3557 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stForm"] input {
        background-color: #F8FAFC !important;
        color: #1D3557 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }

    div[data-testid="stForm"] button[type="submit"] {
        background-color: #0F4C81 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        border: none !important;
    }

    /* =========================================================
       TOMBOL UMUM, UPLOADER & DATAFRAME
    ========================================================= */
    .stButton > button {
        background-color: #0F4C81 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        border: none !important;
        transition: 0.3s !important;
    }

    .stButton > button:hover {
        background-color: #1565C0 !important;
        color: #FFFFFF !important;
        transform: scale(1.01) !important;
    }

    .stDownloadButton > button {
        width: 100% !important;
        background-color: #198754 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: bold !important;
    }

    .stDownloadButton > button:hover {
        background-color: #157347 !important;
    }

    section[data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        padding: 15px !important;
        border-radius: 15px !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.06) !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.06) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #0F4C81;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #ECECEC;
    }
    </style>
    """, unsafe_allow_html=True)


# apply styling
apply_custom_css()


# =========================================================
# INISIALISASI SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "username" not in st.session_state:
    st.session_state["username"] = ""

if "role" not in st.session_state:
    st.session_state["role"] = ""


# =========================================================
# DATABASE AKUN
# =========================================================

USERS = {
    "admin": {
        "password": "admin123",
        "role": "Admin",
        "name": "Administrator"
    },
    "peneliti": {
        "password": "peneliti123",
        "role": "Peneliti",
        "name": "Peneliti"
    }
}


# =========================================================
# HALAMAN LOGIN
# =========================================================

def login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.8, 1])
    
    with col2:
        st.markdown("<h2 style='text-align: center; color: #0F4C81;'>🔐 Login Sistem</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #555555;'>Sistem Clustering K-Means Kesiapan Skripsi</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username_input = st.text_input("Username").strip()
            password_input = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if username_input in USERS and USERS[username_input]["password"] == password_input:
                    user_info = USERS[username_input]
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = user_info["name"]
                    st.session_state["role"] = user_info["role"]
                    st.success(f"✅ Login berhasil sebagai **{user_info['role']}**!")
                    st.rerun()
                else:
                    st.error("❌ Username atau Password salah!")


if not st.session_state["logged_in"]:
    login_page()
    st.stop()


# =========================================================
# SIDEBAR SAMA MENU NAVIGASI
# =========================================================

st.sidebar.title("📊 Clustering K-Means")
st.sidebar.write(f"👤 User: **{st.session_state['username']}**")
st.sidebar.caption(f"🛡️ Role: **{st.session_state['role']}**")

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""
    st.rerun()

st.sidebar.markdown("---")

# akses menu tergantung sama role
if st.session_state["role"] == "Admin":
    menu_options = [
        "🏠 Dashboard",
        "👨‍🎓 Data Mahasiswa",
        "📊 Clustering",
        "📉 Evaluasi Cluster",
        "📈 Hasil Clustering",
        "📄 Laporan"
    ]
else:
    menu_options = [
        "🏠 Dashboard",
        "👨‍🎓 Data Mahasiswa",
        "📊 Clustering",
        "📉 Evaluasi Cluster"
    ]

menu = st.sidebar.radio("Menu", menu_options)


# =========================================================
# DASHBOARD
# =========================================================

if menu == "🏠 Dashboard":

    st.title("📊 Dashboard")

    st.subheader(
        "Sistem Clustering K-Means "
        "Kesiapan Mahasiswa dalam Menyusun Skripsi"
    )

    st.markdown("---")

    if "data_mahasiswa" in st.session_state:
        jumlah_data = len(st.session_state["data_mahasiswa"])
    else:
        jumlah_data = 0

    if "hasil_clustering" in st.session_state:
        jumlah_cluster = st.session_state["hasil_clustering"]["Cluster"].nunique()
        status = "Sudah Diproses"
    else:
        jumlah_cluster = 0
        status = "Belum Diproses"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👨‍🎓 Total Mahasiswa", jumlah_data)

    with col2:
        st.metric("📊 Jumlah Cluster", jumlah_cluster)

    with col3:
        st.metric("📋 Parameter", 4)

    with col4:
        st.metric("🔄 Status", status)

    st.markdown("---")

    st.info(
        "Sistem ini menggunakan algoritma K-Means "
        "untuk mengelompokkan mahasiswa berdasarkan "
        "tingkat kesiapan dalam menyusun skripsi."
    )


# =========================================================
# DATA MAHASISWA
# =========================================================

elif menu == "👨‍🎓 Data Mahasiswa":

    st.title("👨‍🎓 Data Mahasiswa")

    st.write(
        "Upload data hasil kuesioner mahasiswa "
        "dalam format Excel atau CSV."
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📁 Upload Data Kuesioner",
        type=["xlsx", "xls", "csv"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            df.columns = df.columns.astype(str).str.strip()

            st.success(f"✅ File {uploaded_file.name} berhasil dibaca!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("👨‍🎓 Jumlah Responden", len(df))

            with col2:
                st.metric("📋 Jumlah Kolom", len(df.columns))

            with col3:
                st.metric("❓ Data Kosong", int(df.isna().sum().sum()))

            st.markdown("---")

            prokrastinasi = [
                "Saya sering menunda mengerjakan skripsi meskipun memiliki waktu luang",
                "Saya merasa sulit untuk memulai mengerjakan skripsi",
                "Saya lebih memilih melakukan aktivitas lain dibanding mengerjakan skripsi",
                "Saya mengerjakan skripsi hanya ketika mendekati tenggat waktu",
                "Saya sering menunda melakukan bimbingan dengan dosen pembimbing"
            ]

            manajemen_waktu = [
                "Saya memiliki jadwal khusus untuk mengerjakan skripsi",
                "Saya dapat membagi waktu antara kuliah, skripsi, dan aktivitas lain",
                "Saya konsisten mengikuti jadwal pengerjaan skripsi",
                "Saya tetap meluangkan waktu untuk mengerjakan skripsi meskipun memiliki aktivitas lain",
                "Saya mampu menentukan prioritas sehingga penyusunan skripsi tidak tertunda"
            ]

            kesadaran_diri = [
                "Saya menyadari skripsi adalah tanggung jawab utama saya saat ini",
                "Saya memahami konsekuensi jika menunda penyusunan skripsi",
                "Saya menyadari skripsi perlu dikerjakan secara bertahap",
                "Saya bertanggung jawab atas kemajuan skripsi saya sendiri",
                "Saya menyadari pentingnya menyelesaikan skripsi tepat waktu"
            ]

            kepercayaan_diri = [
                "Saya percaya diri dengan kemampuan saya dalam menyusun skripsi",
                "Saya percaya diri saat menjelaskan penelitian saya kepada dosen pembimbing",
                "Saya akan menyusun skripsi dengan optimis agar hasil skripsi sesuai dengan yang diharapkan",
                "Saya yakin dapat menyusun skripsi dengan cepat apabila rajin mengerjakannya",
                "Saya yakin mampu memperbaiki skripsi sesuai arahan dosen pembimbing"
            ]

            semua_pertanyaan = (
                prokrastinasi + manajemen_waktu + kesadaran_diri + kepercayaan_diri
            )

            kolom_normal = {str(kolom).strip().lower(): kolom for kolom in df.columns}
            mapping_kolom = {}

            for pertanyaan in semua_pertanyaan:
                kunci = pertanyaan.strip().lower()
                if kunci in kolom_normal:
                    mapping_kolom[pertanyaan] = kolom_normal[kunci]

            kolom_hilang = [p for p in semua_pertanyaan if p not in mapping_kolom]

            if kolom_hilang:
                st.warning(
                    f"⚠️ Terdapat {len(kolom_hilang)} pertanyaan yang belum cocok dengan nama kolom Excel."
                )
                with st.expander("🔎 Lihat pertanyaan yang belum ditemukan"):
                    for pertanyaan in kolom_hilang:
                        st.write(f"- {pertanyaan}")
                st.info("Periksa kembali nama pertanyaan pada file Excel.")

            else:
                st.success("✅ Semua 20 pertanyaan kuesioner berhasil ditemukan!")

                for pertanyaan in semua_pertanyaan:
                    kolom_asli = mapping_kolom[pertanyaan]
                    df[kolom_asli] = pd.to_numeric(df[kolom_asli], errors="coerce")

                df["Prokrastinasi"] = df[[mapping_kolom[p] for p in prokrastinasi]].mean(axis=1)
                df["Manajemen Waktu"] = df[[mapping_kolom[p] for p in manajemen_waktu]].mean(axis=1)
                df["Kesadaran Diri"] = df[[mapping_kolom[p] for p in kesadaran_diri]].mean(axis=1)
                df["Kepercayaan Diri"] = df[[mapping_kolom[p] for p in kepercayaan_diri]].mean(axis=1)

                st.session_state["data_mahasiswa"] = df.copy()

                st.markdown("---")
                st.subheader("📊 Nilai 4 Parameter")

                kolom_identitas = [k for k in ["Nama", "NIM", "Angkatan", "Program Studi"] if k in df.columns]
                kolom_parameter = ["Prokrastinasi", "Manajemen Waktu", "Kesadaran Diri", "Kepercayaan Diri"]

                tabel_parameter = df[kolom_identitas + kolom_parameter].copy()
                tabel_parameter[kolom_parameter] = tabel_parameter[kolom_parameter].round(2)

                st.dataframe(tabel_parameter, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("📈 Rata-rata Setiap Parameter")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Prokrastinasi", f"{df['Prokrastinasi'].mean():.2f}")
                with col2:
                    st.metric("Manajemen Waktu", f"{df['Manajemen Waktu'].mean():.2f}")
                with col3:
                    st.metric("Kesadaran Diri", f"{df['Kesadaran Diri'].mean():.2f}")
                with col4:
                    st.metric("Kepercayaan Diri", f"{df['Kepercayaan Diri'].mean():.2f}")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")


# =========================================================
# CLUSTERING
# =========================================================

elif menu == "📊 Clustering":

    st.title("📊 Proses K-Means Clustering")
    st.write("Pengelompokan mahasiswa berdasarkan tingkat kesiapan dalam menyusun skripsi.")
    st.markdown("---")

    if "data_mahasiswa" not in st.session_state:
        st.warning("⚠️ Data mahasiswa belum tersedia.")
        st.info("Silakan upload data terlebih dahulu melalui menu Data Mahasiswa.")
        st.stop()

    df = st.session_state["data_mahasiswa"].copy()
    st.success(f"✅ Data tersedia: {len(df)} responden.")

    parameter = ["Prokrastinasi", "Manajemen Waktu", "Kesadaran Diri", "Kepercayaan Diri"]

    st.subheader("📌 Parameter yang Digunakan")
    for i, p in enumerate(parameter, start=1):
        st.write(f"**{i}. {p}**")

    st.markdown("---")

    data_parameter = df[parameter].copy()
    st.subheader("📋 Data Parameter")
    st.dataframe(data_parameter.round(2), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔄 Normalisasi Min-Max")
    st.write("Data dinormalisasi ke rentang 0 sampai 1.")

    scaler = MinMaxScaler()
    data_normalisasi = scaler.fit_transform(data_parameter)

    df_normalisasi = pd.DataFrame(data_normalisasi, columns=parameter, index=df.index)

    st.subheader("🔄 Penyamaan Arah Nilai")
    st.write("Nilai Prokrastinasi dibalik agar seluruh parameter memiliki arah yang sama.")

    df_normalisasi["Prokrastinasi"] = 1 - df_normalisasi["Prokrastinasi"]
    df_normalisasi = df_normalisasi.round(4)

    st.dataframe(df_normalisasi, use_container_width=True, hide_index=True)
    st.success("✅ Normalisasi dan penyamaan arah nilai berhasil.")

    st.session_state["data_normalisasi"] = df_normalisasi.copy()

    st.markdown("---")
    st.subheader("⚙️ Pengaturan K-Means")

    jumlah_cluster = st.number_input("Jumlah Cluster (K)", min_value=2, max_value=10, value=3, step=1)
    jumlah_iterasi = st.number_input("Maksimal Iterasi", min_value=10, max_value=1000, value=300, step=10)

    st.write(f"Jumlah Cluster: **{jumlah_cluster}**")
    st.write(f"Maksimal Iterasi: **{jumlah_iterasi}**")
    st.markdown("---")

    if st.button("🚀 Jalankan K-Means"):
        model = KMeans(n_clusters=jumlah_cluster, max_iter=jumlah_iterasi, random_state=42, n_init=10)
        cluster = model.fit_predict(df_normalisasi[parameter])

        df_hasil = df.copy()
        df_hasil["Cluster"] = cluster + 1

        centroid = pd.DataFrame(model.cluster_centers_, columns=parameter)
        centroid.index = [f"Cluster {i + 1}" for i in range(jumlah_cluster)]
        centroid["Skor Kesiapan"] = centroid[parameter].mean(axis=1)

        centroid_urut = centroid.sort_values(by="Skor Kesiapan", ascending=False)

        label_kesiapan = {}
        for posisi, nama_cluster in enumerate(centroid_urut.index):
            if posisi == 0:
                label_kesiapan[nama_cluster] = "Kesiapan Tinggi"
            elif posisi == jumlah_cluster - 1:
                label_kesiapan[nama_cluster] = "Kesiapan Rendah"
            else:
                label_kesiapan[nama_cluster] = "Kesiapan Sedang"

        df_hasil["Tingkat Kesiapan"] = df_hasil["Cluster"].apply(lambda x: label_kesiapan[f"Cluster {x}"])

        st.session_state["model_kmeans"] = model
        st.session_state["hasil_clustering"] = df_hasil
        st.session_state["centroid"] = centroid
        st.session_state["label_kesiapan"] = label_kesiapan

        st.success("🎉 K-Means berhasil dijalankan!")
        st.markdown("---")

        st.subheader("📊 Hasil Pengelompokan")
        kolom_identitas = [k for k in ["Nama", "NIM", "Angkatan", "Program Studi"] if k in df_hasil.columns]
        tabel_hasil = df_hasil[kolom_identitas + parameter + ["Cluster", "Tingkat Kesiapan"]].copy()
        tabel_hasil[parameter] = tabel_hasil[parameter].round(2)

        st.dataframe(tabel_hasil, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("👥 Jumlah Mahasiswa Setiap Tingkat Kesiapan")

        jumlah_anggota = df_hasil["Tingkat Kesiapan"].value_counts()
        urutan_label = ["Kesiapan Tinggi", "Kesiapan Sedang", "Kesiapan Rendah"]
        label_tersedia = [label for label in urutan_label if label in jumlah_anggota.index]

        cols = st.columns(len(label_tersedia))
        for i, label in enumerate(label_tersedia):
            with cols[i]:
                st.metric(label, f"{jumlah_anggota[label]} mahasiswa")

        st.markdown("---")
        st.subheader("📍 Centroid Akhir")

        centroid_tampil = centroid.copy()
        centroid_tampil["Tingkat Kesiapan"] = centroid_tampil.index.map(label_kesiapan)
        centroid_tampil = centroid_tampil[parameter + ["Skor Kesiapan", "Tingkat Kesiapan"]].round(4)

        st.dataframe(centroid_tampil, use_container_width=True)

        st.markdown("---")
        st.subheader("🏆 Urutan Tingkat Kesiapan")

        for posisi, (nama_cluster, baris) in enumerate(centroid_urut.iterrows(), start=1):
            label = label_kesiapan[nama_cluster]
            st.write(f"**{posisi}. {nama_cluster} → {label}** (Skor: {baris['Skor Kesiapan']:.4f})")

        st.info("Nomor cluster ditentukan oleh algoritma K-Means dan tidak menunjukkan tingkat kesiapan secara langsung.")


# =========================================================
# EVALUASI CLUSTER
# =========================================================

elif menu == "📉 Evaluasi Cluster":

    st.title("📉 Evaluasi Cluster")

    if "data_normalisasi" not in st.session_state:
        st.warning("Silakan lakukan clustering terlebih dahulu.")
        st.stop()

    data = st.session_state["data_normalisasi"]

    st.subheader("Elbow Method")

    k_range = range(2, 8)
    wcss = []
    silhouette = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        label = model.fit_predict(data)
        wcss.append(model.inertia_)
        silhouette.append(silhouette_score(data, label))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), wcss, marker="o")
    ax.set_xlabel("Jumlah Cluster (K)")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")

    st.pyplot(fig)
    st.markdown("---")

    st.subheader("Silhouette Score")

    hasil = pd.DataFrame({
        "Jumlah Cluster": list(k_range),
        "Silhouette Score": [round(x, 4) for x in silhouette]
    })

    st.dataframe(hasil, use_container_width=True, hide_index=True)

    terbaik = hasil.loc[hasil["Silhouette Score"].idxmax()]

    st.success(
        f"Cluster terbaik berdasarkan Silhouette Score adalah K = {int(terbaik['Jumlah Cluster'])} "
        f"dengan nilai {terbaik['Silhouette Score']:.4f}"
    )

    st.info("Semakin tinggi Silhouette Score maka kualitas cluster semakin baik.")


# =========================================================
# HASIL CLUSTERING
# =========================================================

elif menu == "📈 Hasil Clustering":

    st.title("📈 Hasil Clustering")

    if "hasil_clustering" not in st.session_state:
        st.warning("⚠️ K-Means belum dijalankan.")
        st.info("Silakan masuk ke menu Clustering dan jalankan K-Means terlebih dahulu.")
        st.stop()

    df_hasil = st.session_state["hasil_clustering"].copy()

    st.subheader("📊 Ringkasan Hasil")

    jumlah_tinggi = len(df_hasil[df_hasil["Tingkat Kesiapan"] == "Kesiapan Tinggi"])
    jumlah_sedang = len(df_hasil[df_hasil["Tingkat Kesiapan"] == "Kesiapan Sedang"])
    jumlah_rendah = len(df_hasil[df_hasil["Tingkat Kesiapan"] == "Kesiapan Rendah"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🟢 Kesiapan Tinggi", jumlah_tinggi)
    with col2:
        st.metric("🟡 Kesiapan Sedang", jumlah_sedang)
    with col3:
        st.metric("🔴 Kesiapan Rendah", jumlah_rendah)

    st.markdown("---")

    st.subheader("📋 Data Hasil Clustering")

    kolom_tampil = [
        k for k in [
            "Nama", "NIM", "Angkatan", "Program Studi",
            "Prokrastinasi", "Manajemen Waktu", "Kesadaran Diri", "Kepercayaan Diri",
            "Cluster", "Tingkat Kesiapan"
        ] if k in df_hasil.columns
    ]

    st.dataframe(df_hasil[kolom_tampil].round(2), use_container_width=True, hide_index=True)

    if "centroid" in st.session_state:
        st.markdown("---")
        st.subheader("📍 Centroid Cluster")

        centroid = st.session_state["centroid"].copy()
        label_kesiapan = st.session_state["label_kesiapan"]
        centroid["Tingkat Kesiapan"] = centroid.index.map(label_kesiapan)

        st.dataframe(centroid.round(4), use_container_width=True)


# =========================================================
# LAPORAN
# =========================================================

elif menu == "📄 Laporan":

    st.title("📄 Laporan Hasil Clustering")

    if "hasil_clustering" not in st.session_state:
        st.warning("⚠️ Belum ada hasil clustering.")
        st.info("Jalankan K-Means terlebih dahulu.")
        st.stop()

    df_hasil = st.session_state["hasil_clustering"].copy()

    st.subheader("📊 Ringkasan")

    total = len(df_hasil)
    tinggi = len(df_hasil[df_hasil["Tingkat Kesiapan"] == "Kesiapan Tinggi"])
    sedang = len(df_hasil[df_hasil["Tingkat Kesiapan"] == "Kesiapan Sedang"])
    rendah = len(df_hasil[df_hasil["Tingkat Kesiapan"] == "Kesiapan Rendah"])

    st.write(f"Total mahasiswa: **{total}**")
    st.write(f"Kesiapan tinggi: **{tinggi} mahasiswa**")
    st.write(f"Kesiapan sedang: **{sedang} mahasiswa**")
    st.write(f"Kesiapan rendah: **{rendah} mahasiswa**")

    st.markdown("---")

    st.subheader("📋 Data Hasil Clustering")

    st.dataframe(df_hasil, use_container_width=True, hide_index=True)

    csv = df_hasil.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Hasil CSV",
        data=csv,
        file_name="hasil_clustering_kmeans.csv",
        mime="text/csv"
    )