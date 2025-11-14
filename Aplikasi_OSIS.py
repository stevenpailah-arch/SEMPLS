import streamlit as st
import pandas as pd
import json
from PIL import Image
import os

# --- KONSTANTA DAN KONFIGURASI ---
DATA_FILE = 'results.json'
PASSWORD = "123456" # Kata sandi untuk akses admin (input, reset, konfigurasi)

# Data kandidat awal (hanya digunakan jika file data belum ada)
# Catatan: Jika Anda menambahkan kandidat baru di konfigurasi, pastikan foto sudah tersedia!
INITIAL_CANDIDATES = {
    "Nabil (Ketua OSIS)": {"votes": 0, "image": "nabil.jpg"},
    "Siska (Wakil Ketua)": {"votes": 0, "image": "siska.jpg"},
    "Reno (Bendahara)": {"votes": 0, "image": "reno.jpg"}
}

# --- FUNGSI UNTUK PERSISTENSI DATA (JSON) ---

def load_data():
    """Memuat data dari file JSON."""
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            # Memastikan struktur data lengkap
            for name, details in INITIAL_CANDIDATES.items():
                if name not in data and name == list(INITIAL_CANDIDATES.keys())[0]: # Hanya untuk inisialisasi awal
                    return INITIAL_CANDIDATES
            return data
    except FileNotFoundError:
        save_data(INITIAL_CANDIDATES)
        return INITIAL_CANDIDATES
    except json.JSONDecodeError:
        st.error("Error: File data rusak. Menggunakan data awal.")
        save_data(INITIAL_CANDIDATES)
        return INITIAL_CANDIDATES

def save_data(data):
    """Menyimpan data ke file JSON."""
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        st.session_state.results = data # Update session state setelah save
    except Exception as e:
        st.error(f"Gagal menyimpan data ke file: {e}")

def reset_data():
    """Mengatur ulang data pemilihan ke kondisi awal (hanya suara)."""
    current_data = load_data()
    # Reset hanya jumlah suara, nama kandidat tetap
    for name in current_data:
        current_data[name]['votes'] = 0
    save_data(current_data)
    st.success("Data pemilihan berhasil direset ke nol!")

# --- INISIALISASI SESSION STATE ---
if 'results' not in st.session_state:
    st.session_state.results = load_data()

# --- KONFIGURASI TAMPILAN (Dark Green Theme) ---

st.markdown(
    """
    <style>
    /* Tema Dark Green */
    .stApp {
        background-color: #121212; 
        color: #e0e0e0; 
    }
    .stButton>button {
        background-color: #004d40;
        color: white;
        border-radius: 8px;
        border: 1px solid #004d40;
    }
    .stButton>button:hover {
        background-color: #00796b;
        border: 1px solid #00796b;
    }
    h1 {
        color: #009688;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🗳️ Aplikasi Pemilihan Ketua OSIS")

# --- FUNGSI UNTUK MENGINPUT SUARA (Link Input) ---

def voting_page():
    st.header("Halaman Input Suara (Admin)")
    
    st.markdown("---")
    st.subheader("Akses Admin")
    
    password_input = st.text_input("Masukkan Kata Sandi:", type="password", key='voting_password')
    
    if password_input == PASSWORD:
        st.success("Akses Diterima! Silakan catat suara.")
        
        candidates = st.session_state.results.keys()
        total_votes = sum(item['votes'] for item in st.session_state.results.values())
        st.info(f"Total Suara Tercatat Saat Ini: **{total_votes}**")
        
        with st.form(key='vote_form'):
            selected_candidate = st.radio(
                "Pilih Kandidat yang Menerima Suara:",
                options=list(candidates),
                key='candidate_choice'
            )
            
            submit_button = st.form_submit_button(label='✅ Catat Suara')
            
            if submit_button:
                # Logika voting
                st.session_state.results[selected_candidate]['votes'] += 1
                
                # Simpan data permanen setelah voting
                save_data(st.session_state.results)
                
                # Tambahan: Pemberitahuan dengan jumlah suara yang tercatat
                new_total_votes = sum(item['votes'] for item in st.session_state.results.values())
                st.success(f"Suara berhasil dicatat untuk **{selected_candidate}**. Total suara saat ini: **{new_total_votes}**.")
                # st.rerun() # Tidak perlu rerun jika hanya menampilkan success message
                
    elif password_input: 
        st.error("Kata Sandi Salah. Akses Ditolak.")

# --- FUNGSI UNTUK KONFIGURASI KANDIDAT ---

def config_page():
    st.header("⚙️ Konfigurasi Kandidat (Admin)")
    
    st.markdown("---")
    st.subheader("Akses Konfigurasi")
    
    password_input = st.text_input("Masukkan Kata Sandi:", type="password", key='config_password')
    
    if password_input == PASSWORD:
        st.success("Akses Diterima! Ubah data kandidat di bawah.")
        
        # Menggunakan tab untuk memisahkan konfigurasi tiap kandidat
        candidate_names = list(st.session_state.results.keys())
        
        tabs = st.tabs(candidate_names)
        
        new_data = dict(st.session_state.results) # Buat salinan untuk perubahan
        
        for i, name in enumerate(candidate_names):
            with tabs[i]:
                current_image = new_data[name]['image']
                
                st.markdown(f"### Data Kandidat: **{name}**")
                
                # Input Nama Baru
                new_name = st.text_input("Nama Kandidat Baru:", value=name, key=f"name_{name}")
                
                # Input File Foto
                uploaded_file = st.file_uploader(
                    f"Upload Foto Baru (Saat ini: **{current_image}**):", 
                    type=["jpg", "jpeg", "png"],
                    key=f"upload_{name}"
                )

                st.info(f"File foto kandidat saat ini: `{current_image}`")
                
                # Tombol Simpan Perubahan
                if st.button(f"Simpan Perubahan untuk {name}", key=f"save_{name}"):
                    
                    # 1. Menangani Perubahan Foto
                    if uploaded_file is not None:
                        # Simpan file yang diupload
                        file_extension = uploaded_file.name.split('.')[-1]
                        new_image_filename = f"{new_name.replace(' ', '_').lower()}.{file_extension}"
                        
                        # Simpan file ke disk
                        with open(os.path.join("./", new_image_filename), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        new_data[name]['image'] = new_image_filename
                        st.success(f"Foto berhasil diupdate menjadi `{new_image_filename}`.")
                    
                    # 2. Menangani Perubahan Nama
                    if new_name != name:
                        # Jika nama berubah, kita harus membuat entri baru dan menghapus yang lama
                        new_data[new_name] = new_data.pop(name)
                        
                        # Pastikan kunci 'image' yang baru juga diupdate jika ada upload
                        if uploaded_file is None:
                             # Jika tidak ada upload, pastikan nama file image tetap di kunci baru
                             new_data[new_name]['image'] = current_image

                        st.success(f"Nama berhasil diubah dari '{name}' menjadi '{new_name}'.")

                    # Simpan data permanen dan update session state
                    save_data(new_data)
                    st.toast("Konfigurasi disimpan, me-reload halaman...")
                    st.rerun()
                    
    elif password_input:
        st.error("Kata Sandi Salah. Akses Ditolak.")

# --- FUNGSI UNTUK MENAMPILKAN HASIL ---

def results_page():
    # ... (Fungsi results_page sama seperti sebelumnya) ...
    st.header("Tampilan Hasil Pemilihan Saat Ini")
    
    current_results = st.session_state.results
    data = []
    total_votes = sum(item['votes'] for item in current_results.values())

    # Konversi ke data frame
    for name, item in current_results.items():
        votes = item['votes']
        percentage = (votes / total_votes * 100) if total_votes > 0 else 0
        data.append([name, votes, f"{percentage:.2f}%"])

    df = pd.DataFrame(data, columns=['Kandidat', 'Jumlah Suara', 'Persentase'])
    df = df.sort_values(by='Jumlah Suara', ascending=False).reset_index(drop=True)

    st.subheader(f"Total Suara Masuk: {total_votes}")
    
    st.table(df)

    st.markdown("---")

    st.subheader("Rincian Hasil dan Foto Kandidat")
    
    cols = st.columns(len(current_results))
    
    for i, (name, item) in enumerate(current_results.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            
            try:
                img = Image.open(item['image'])
                st.image(img, caption=name, use_column_width=True)
            except FileNotFoundError:
                st.error(f"Foto '{item['image']}' tidak ditemukan. Harap unggah foto baru di menu Konfigurasi.")
            except Exception:
                 st.info(f"Tidak dapat memuat foto.")

            st.metric(label="Suara", value=item['votes'])


# --- FUNGSI UNTUK RESET DATA (Dilindungi Password) ---

def reset_page():
    st.sidebar.markdown("---")
    st.sidebar.subheader("Area Reset Data (Admin)")
    
    # Reset Button
    reset_password = st.sidebar.text_input("Kata Sandi Reset:", type="password", key='reset_password')
    
    if reset_password == PASSWORD:
        st.sidebar.warning("PERINGATAN: Tindakan ini hanya mereset jumlah suara menjadi nol.")
        if st.sidebar.button("🚨 RESET SEMUA SUARA 🚨"):
            reset_data()
            st.rerun()
    elif reset_password:
        st.sidebar.error("Kata Sandi Salah.")

# --- NAVIGASI UTAMA ---

# Sidebar untuk navigasi
st.sidebar.title("Navigasi Aplikasi")
page = st.sidebar.radio("Pilih Tampilan:", [
    "Tampilan Hasil", 
    "Input Suara", 
    "Konfigurasi Kandidat"
])

if page == "Tampilan Hasil":
    results_page()
elif page == "Input Suara":
    voting_page()
elif page == "Konfigurasi Kandidat":
    config_page()

# Tambahkan fitur reset data di sidebar
reset_page()

# Tambahkan informasi di sidebar
st.sidebar.markdown("---")
st.sidebar.caption("Kata Sandi Admin: **123456**")