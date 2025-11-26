import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from io import StringIO

# --- Header Profil ---
st.markdown("""
    <div style="display: flex; align-items: center; gap: 20px;">
        <img src="https://scholar.googleusercontent.com/citations?view_op=view_photo&user=lpq1xBMAAAAJ&citpid=1"
             alt="Foto Profil"
             style="width: 130px; height: 130px; border-radius: 50%; object-fit: cover; border: 3px solid #4CAF50;">
        <div>
            <h2 style="margin-bottom: 5px;">Fatrisye Pandensolang</h2>
            <p style="margin: 0; font-size: 18px;">BNN Provinsi Sulawesi Utara</p>
            <p style="margin: 0; font-size: 18px;">E-mail: <b>fatrisye@gmail.com</b></p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Konfigurasi Dasar ---
st.set_page_config(
    page_title="Aplikasi Analisis SEM PLS Proxy",
    layout="wide"
)

# --- Definisi Variabel dan Indikator ---
INDICATORS = {
    'X1': {'name': 'Kompetensi', 'cols': ['X1.1', 'X1.2', 'X1.3', 'X1.4', 'X1.5']},
    'X2': {'name': 'Motivasi', 'cols': ['X2.1', 'X2.2', 'X2.3', 'X2.4', 'X2.5']},
    'X3': {'name': 'Kepemimpinan', 'cols': ['X3.1', 'X3.2', 'X3.3', 'X3.4', 'X3.5']},
    'X4': {'name': 'Lingkungan Kerja', 'cols': ['X4.1', 'X4.2', 'X4.3', 'X4.4', 'X4.5']},
    'Y': {'name': 'Kepuasan Kerja', 'cols': ['Y1.1', 'Y2.2', 'Y3.3', 'Y4.4', 'Y5.5']},
    'Z': {'name': 'Kinerja Pegawai', 'cols': ['Z1.1', 'Z1.2', 'Z1.3', 'Z1.4', 'Z1.5']}
}

# Fungsi untuk melakukan analisis
def analyze_data(df, threshold_validity=0.5, alpha=0.05):
    all_indicator_cols = [col for key in INDICATORS for col in INDICATORS[key]['cols']]
    
    # 1. Pembersihan Data
    df_clean = df.copy()
    for col in all_indicator_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean.dropna(subset=all_indicator_cols, inplace=True)
    
    if df_clean.empty:
        return "Error: Tidak ada data yang tersisa setelah membersihkan nilai yang hilang.", None

    # 2. Uji Validitas (Item-Total Correlation)
    validity_results = {}
    valid_indicators = {key: {'name': val['name'], 'cols': []} for key, val in INDICATORS.items()}
    invalid_indicators_list = []

    # Hitung skor laten awal (menggunakan semua item untuk korelasi awal)
    df_scores_all = pd.DataFrame()
    for var, data in INDICATORS.items():
        df_scores_all[var] = df_clean[data['cols']].mean(axis=1)

    for var, data in INDICATORS.items():
        validity_results[var] = []
        for col in data['cols']:
            # Pastikan kolom ada di df_clean
            if col in df_clean.columns:
                correlation = df_clean[col].corr(df_scores_all[var])
                valid = correlation >= threshold_validity
                validity_results[var].append({
                    'Item': col,
                    'Korelasi (r)': correlation,
                    'Status': 'VALID' if valid else 'TIDAK VALID'
                })
                if valid:
                    valid_indicators[var]['cols'].append(col)
                else:
                    invalid_indicators_list.append(col)
    
    # 3. Rekalkulasi Skor Laten (Hanya menggunakan indikator yang valid)
    df_scores_valid = pd.DataFrame()
    for var, data in valid_indicators.items():
        if not data['cols']:
            return f"Error: Variabel {data['name']} tidak memiliki indikator yang valid.", None
        df_scores_valid[var] = df_clean[data['cols']].mean(axis=1)

    # 4. Path Analysis (SEM PLS Proxy - OLS Regression)
    
    # Model 1: X -> Y
    X_model1 = df_scores_valid[['X1', 'X2', 'X3', 'X4']]
    y_model1 = df_scores_valid['Y']
    X_model1 = sm.add_constant(X_model1)
    model1 = sm.OLS(y_model1, X_model1).fit()

    # Model 2: X, Y -> Z
    X_model2 = df_scores_valid[['X1', 'X2', 'X3', 'X4', 'Y']]
    y_model2 = df_scores_valid['Z']
    X_model2 = sm.add_constant(X_model2)
    model2 = sm.OLS(y_model2, X_model2).fit()
    
    # 5. Penghitungan Efek
    
    # Koefisien Jalur
    beta_X1Y = model1.params['X1']
    beta_X2Y = model1.params['X2']
    beta_X3Y = model1.params['X3']
    beta_X4Y = model1.params['X4']
    beta_YZ = model2.params['Y']
    
    # P-value Jalur Langsung ke Y
    p_X1Y = model1.pvalues['X1']
    p_X2Y = model1.pvalues['X2']
    p_X3Y = model1.pvalues['X3']
    p_X4Y = model1.pvalues['X4']
    
    # P-value Jalur Langsung ke Z (ambil dari model2)
    p_X1Z = model2.pvalues['X1']
    p_X2Z = model2.pvalues['X2'] # Diperbaiki, sebelumnya salah ambil X3 dan X4
    p_X3Z = model2.pvalues['X3'] # Diperbaiki
    p_X4Z = model2.pvalues['X4'] # Diperbaiki
    p_YZ = model2.pvalues['Y']

    # Efek Tidak Langsung (Indirect Effects)
    indirect_X1Z = beta_X1Y * beta_YZ
    indirect_X2Z = beta_X2Y * beta_YZ
    indirect_X3Z = beta_X3Y * beta_YZ
    indirect_X4Z = beta_X4Y * beta_YZ

    # Efek Langsung (Direct Effects)
    direct_X1Z = model2.params['X1']
    direct_X2Z = model2.params['X2']
    direct_X3Z = model2.params['X3']
    direct_X4Z = model2.params['X4']
    
    # Efek Total (Total Effects)
    total_X1Z = direct_X1Z + indirect_X1Z
    total_X2Z = direct_X2Z + indirect_X2Z
    total_X3Z = direct_X3Z + indirect_X3Z
    total_X4Z = direct_X4Z + indirect_X4Z
    
    # 6. Ringkasan Hasil Analisis Struktural
    
    results = {
        'model1': model1, # Objek model OLS
        'model2': model2, # Objek model OLS
        'validity': validity_results,
        'invalid_items': invalid_indicators_list,
        'valid_indicators': valid_indicators,
        'path_analysis': {
            # H1-H4
            'X1Y': {'beta': beta_X1Y, 'p': p_X1Y},
            'X2Y': {'beta': beta_X2Y, 'p': p_X2Y},
            'X3Y': {'beta': beta_X3Y, 'p': p_X3Y},
            'X4Y': {'beta': beta_X4Y, 'p': p_X4Y},
            # H5-H8 & H9 (Direct)
            'X1Z': {'beta': direct_X1Z, 'p': p_X1Z},
            'X2Z': {'beta': direct_X2Z, 'p': p_X2Z},
            'X3Z': {'beta': direct_X3Z, 'p': p_X3Z},
            'X4Z': {'beta': direct_X4Z, 'p': p_X4Z},
            'YZ': {'beta': beta_YZ, 'p': p_YZ},
            # H10-H13 (Indirect)
            'X1YZ': indirect_X1Z,
            'X2YZ': indirect_X2Z,
            'X3YZ': indirect_X3Z,
            'X4YZ': indirect_X4Z,
            # Total
            'X1Z_total': total_X1Z,
            'X2Z_total': total_X2Z,
            'X3Z_total': total_X3Z,
            'X4Z_total': total_X4Z,
        }
    }
    
    return "Analisis Selesai", results

# --- Fungsi Tampilan Laporan ---

def display_report(results, alpha):
    st.header("3. Laporan Hasil Analisis SEM PLS (Proxy)")
    
    # Tampilkan Hasil Uji Validitas
    st.subheader("3.1. Hasil Uji Validitas Konvergen (Item-Total Correlation)")
    st.info(f"Ambangan Batas Korelasi Item-Total (Proksi Outer Loading): **r ≥ {results['threshold']:.2f}**")
    
    validity_data = []
    for var_key, items in results['validity'].items():
        var_name = INDICATORS[var_key]['name']
        for item in items:
            validity_data.append({
                'Variabel': f"{var_key} ({var_name})",
                'Indikator': item['Item'],
                'Korelasi (r)': f"{item['Korelasi (r)']:.4f}",
                'Status': item['Status']
            })
    
    df_validity = pd.DataFrame(validity_data)
    st.dataframe(df_validity, use_container_width=True)
    
    if results['invalid_items']:
        st.warning(f"**Indikator yang Dihapus:** {', '.join(results['invalid_items'])}. Analisis Struktural menggunakan indikator yang tersisa.")
    else:
        st.success("Semua indikator dinyatakan **VALID** (r ≥ 0.50). Analisis Struktural menggunakan semua indikator.")

    # Tampilkan R-Square dan VIF
    st.subheader("3.2. Evaluasi Model Struktural ($\mathbf{R}^2$)")
    
    # PERBAIKAN: Mengakses langsung 'model1' dan 'model2' di root dictionary results
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"R-Square (Kepuasan Kerja / Y)", value=f"{results['model1'].rsquared:.3f}")
        st.caption(f"Kontribusi X1, X2, X3, X4 terhadap Y adalah {results['model1'].rsquared*100:.1f}%.")
    with col2:
        st.metric(label=f"R-Square (Kinerja Pegawai / Z)", value=f"{results['model2'].rsquared:.3f}")
        st.caption(f"Kontribusi X1, X2, X3, X4, Y terhadap Z adalah {results['model2'].rsquared*100:.1f}%.")
    
    # Tampilkan Ringkasan Hipotesa
    st.subheader("3.3. Ringkasan Pengujian Hipotesa (Efek Langsung & Tidak Langsung)")

    # Data Hipotesa Langsung ke Y (H1-H4)
    data_Y = [
        ('H1', 'X1 (Kompetensi) → Y (Kepuasan Kerja)', results['path_analysis']['X1Y']['beta'], results['path_analysis']['X1Y']['p']),
        ('H2', 'X2 (Motivasi) → Y (Kepuasan Kerja)', results['path_analysis']['X2Y']['beta'], results['path_analysis']['X2Y']['p']),
        ('H3', 'X3 (Kepemimpinan) → Y (Kepuasan Kerja)', results['path_analysis']['X3Y']['beta'], results['path_analysis']['X3Y']['p']),
        ('H4', 'X4 (Lingkungan Kerja) → Y (Kepuasan Kerja)', results['path_analysis']['X4Y']['beta'], results['path_analysis']['X4Y']['p']),
    ]
    
    # Data Hipotesa Langsung ke Z (H5-H9)
    data_Z_direct = [
        ('H5', 'X1 (Kompetensi) → Z (Kinerja Pegawai)', results['path_analysis']['X1Z']['beta'], results['path_analysis']['X1Z']['p']),
        ('H6', 'X2 (Motivasi) → Z (Kinerja Pegawai)', results['path_analysis']['X2Z']['beta'], results['path_analysis']['X2Z']['p']),
        ('H7', 'X3 (Kepemimpinan) → Z (Kinerja Pegawai)', results['path_analysis']['X3Z']['beta'], results['path_analysis']['X3Z']['p']),
        ('H8', 'X4 (Lingkungan Kerja) → Z (Kinerja Pegawai)', results['path_analysis']['X4Z']['beta'], results['path_analysis']['X4Z']['p']),
        ('H9', 'Y (Kepuasan Kerja) → Z (Kinerja Pegawai)', results['path_analysis']['YZ']['beta'], results['path_analysis']['YZ']['p']),
    ]
    
    # Data Hipotesa Tidak Langsung (H10-H13)
    data_Z_indirect = [
        ('H10', 'X1 → Y → Z', results['path_analysis']['X1YZ']),
        ('H11', 'X2 → Y → Z', results['path_analysis']['X2YZ']),
        ('H12', 'X3 → Y → Z', results['path_analysis']['X3YZ']),
        ('H13', 'X4 → Y → Z', results['path_analysis']['X4YZ']),
    ]

    # Gabungan Hasil Hipotesa
    final_report_data = []
    
    # Loop H1-H9 (Langsung)
    for h, desc, beta, p in (data_Y + data_Z_direct):
        significance = p <= alpha
        keputusan = "DITERIMA" if significance else "DITOLAK"
        keterangan = f"Signifikan ({p:.4f} ≤ {alpha})" if significance else f"Tidak Signifikan ({p:.4f} > {alpha})"
        final_report_data.append({
            'Hipotesa': h,
            'Jalur': desc,
            'Koef. Jalur (β)': f"{beta:.4f}",
            'P-value': f"{p:.4f}",
            'Keputusan': keputusan,
            'Keterangan': keterangan,
            'Jenis Efek': 'Langsung',
            'Sifat Pengaruh': 'Positif' if beta > 0 and significance else ('Negatif' if beta < 0 and significance else 'Tidak Signifikan')
        })
        
    # Loop H10-H13 (Tidak Langsung)
    for h, desc, indirect_effect in data_Z_indirect:
        key_X = desc.split("→")[0].strip()
        
        # P-value untuk jalur a (X -> Y) dan jalur b (Y -> Z)
        p_XY = results['path_analysis'][f'{key_X}Y']['p']
        p_YZ = results['path_analysis']['YZ']['p']
        
        # P-value untuk jalur c' (Langsung X -> Z)
        direct_p = results['path_analysis'][f'{key_X}Z']['p']

        # Kriteria Mediasi
        is_a_sig = p_XY <= alpha
        is_b_sig = p_YZ <= alpha
        is_c_prime_sig = direct_p <= alpha
        
        keputusan = "DITOLAK"
        keterangan = "Mediasi Tidak Terjadi"

        if is_a_sig and is_b_sig:
            if not is_c_prime_sig:
                keputusan = "DITERIMA"
                keterangan = "Mediasi Penuh (Full Mediation)"
            else:
                keputusan = "DITERIMA"
                keterangan = "Mediasi Parsial (Partial Mediation)"
        
        final_report_data.append({
            'Hipotesa': h,
            'Jalur': desc,
            'Koef. Jalur (β)': f"{indirect_effect:.4f}",
            'P-value': 'N/A',
            'Keputusan': keputusan,
            'Keterangan': keterangan,
            'Jenis Efek': 'Tidak Langsung',
            'Sifat Pengaruh': 'Positif' if indirect_effect > 0 else 'Negatif'
        })

    df_final_report = pd.DataFrame(final_report_data)
    st.dataframe(df_final_report, use_container_width=True)
    
    # Tabel Efek Total
    st.subheader("3.4. Ringkasan Efek Total")
    
    df_total_effects = pd.DataFrame({
        'Jalur': ['X1 (Kompetensi) → Z', 'X2 (Motivasi) → Z', 'X3 (Kepemimpinan) → Z', 'X4 (Lingkungan Kerja) → Z'],
        'Koef. Langsung (Direct)': [
            results['path_analysis']['X1Z']['beta'], 
            results['path_analysis']['X2Z']['beta'], 
            results['path_analysis']['X3Z']['beta'], 
            results['path_analysis']['X4Z']['beta']
        ],
        'Koef. Tidak Langsung (Indirect)': [
            results['path_analysis']['X1YZ'],
            results['path_analysis']['X2YZ'],
            results['path_analysis']['X3YZ'],
            results['path_analysis']['X4YZ']
        ],
        'Koef. Total (Total)': [
            results['path_analysis']['X1Z_total'],
            results['path_analysis']['X2Z_total'],
            results['path_analysis']['X3Z_total'],
            results['path_analysis']['X4Z_total']
        ]
    }).set_index('Jalur').applymap(lambda x: f'{x:.4f}')
    
    st.dataframe(df_total_effects, use_container_width=True)

    st.markdown("---")
    st.markdown("**Catatan:** Analisis ini menggunakan Proksi SEM PLS dengan regresi OLS pada skor rata-rata variabel laten. Nilai $\\beta$ pada Efek Tidak Langsung adalah perkalian $\\beta_{X\\rightarrow Y} \\times \\beta_{Y\\rightarrow Z}$ dan signifikansi ditentukan berdasarkan analisis jalur $\\text{X}\\rightarrow\\text{Y}$ dan $\\text{Y}\\rightarrow\\text{Z}$ (Mediasi Full/Partial).")


# --- Tampilan Streamlit ---

st.title("Aplikasi Analisis SEM PLS (Proxy) dengan Python")
st.markdown("""
Aplikasi ini melakukan analisis pengaruh struktural menggunakan pendekatan Path Analysis (Proksi SEM PLS)
berdasarkan model: $\mathbf{X} \\rightarrow \mathbf{Y}$ dan $\mathbf{X}, \mathbf{Y} \\rightarrow \mathbf{Z}$.

**Definisi Variabel:**
* $\mathbf{X1}$: Kompetensi
* $\mathbf{X2}$: Motivasi
* $\mathbf{X3}$: Kepemimpinan
* $\mathbf{X4}$: Lingkungan Kerja
* $\mathbf{Y}$: Kepuasan Kerja (Intervening)
* $\mathbf{Z}$: Kinerja Pegawai (Dependent)
""")

st.header("1. Upload Data")
uploaded_file = st.file_uploader("Unggah file data kuesioner Anda (.csv)", type="csv")

if uploaded_file is not None:
    try:
        # Menggunakan delimiter ';' sesuai data sebelumnya
        df = pd.read_csv(uploaded_file, delimiter=';')
        
        # Validasi kolom yang dibutuhkan
        all_required_cols = [col for key in INDICATORS for col in INDICATORS[key]['cols']]
        missing_cols = [col for col in all_required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"File tidak memiliki kolom yang diperlukan untuk analisis. Kolom yang hilang: {', '.join(missing_cols)}")
        else:
            st.success("File berhasil diunggah dan diverifikasi.")
            st.dataframe(df.head())

            st.header("2. Pengaturan Analisis")
            col_validity, col_alpha = st.columns(2)
            with col_validity:
                threshold = st.slider(
                    "Ambangan Batas Validitas (r, Proksi Outer Loading)", 
                    min_value=0.3, max_value=0.7, value=0.5, step=0.05
                )
            with col_alpha:
                alpha = st.slider(
                    "Tingkat Signifikansi ($\mathbf{\\alpha}$)", 
                    min_value=0.01, max_value=0.1, value=0.05, step=0.01
                )
            
            if st.button("Jalankan Analisis SEM PLS (Proxy)"):
                st.markdown("---")
                with st.spinner('Sedang melakukan pengujian validitas dan analisis jalur...'):
                    status, results = analyze_data(df, threshold_validity=threshold, alpha=alpha)
                    
                    if status.startswith("Error"):
                        st.error(status)
                    else:
                        results['threshold'] = threshold
                        results['alpha'] = alpha
                        display_report(results, alpha)

    except Exception as e:
        # Menangkap semua kesalahan umum selama pemrosesan data
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.info("Pastikan file CSV menggunakan titik koma (;) sebagai delimiter dan semua kolom indikator terisi dengan angka.")
else:

    st.info("Silakan unggah file CSV Anda untuk memulai analisis.")
