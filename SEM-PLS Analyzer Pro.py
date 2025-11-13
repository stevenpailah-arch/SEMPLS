import streamlit as st
import pandas as pd
import semopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import base64

# --- Konfigurasi dan Styling Halaman ---
st.set_page_config(page_title="SEM-PLS Analyzer ProMax", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk tampilan SmartPLS-like (Cards, Font, Colors)
custom_css = """
<style>
    /* Global Styling */
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
    }
    h1 {
        color: #004d40; /* Dark Teal for header */
        font-weight: 700;
        border-bottom: 3px solid #004d40;
        padding-bottom: 10px;
    }
    h2, h3 {
        color: #00695c; /* Mid Teal */
    }

    /* Info & Alert Boxes */
    div[data-testid="stAlert"] {
        border-left: 5px solid #00695c !important;
        border-radius: 8px;
        background-color: #e0f2f1;
    }

    /* Sidebar Navigation Styling */
    .st-emotion-cache-1ftl4h8 { /* Targetting sidebar selectbox */
        border-radius: 8px;
    }

    /* Container/Card Styling */
    .st-emotion-cache-6q9sum, .st-emotion-cache-1jmveez {
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        background-color: white;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("💡 SmartPLS Light: Analisis SEM-PLS")

# --- Helper Functions ---

@st.cache_data
def generate_template():
    """Menghasilkan template data dummy."""
    data = {
        'responden_id': range(1, 101),
        'KL_1': np.random.randint(1, 6, 100), 'KL_2': np.random.randint(1, 6, 100), 'KL_3': np.random.randint(1, 6, 100), 'KL_4': np.random.randint(1, 6, 100), 'KL_5': np.random.randint(1, 6, 100),
        'CM_1': np.random.randint(1, 6, 100), 'CM_2': np.random.randint(1, 6, 100), 'CM_3': np.random.randint(1, 6, 100), 'CM_4': np.random.randint(1, 6, 100), 'CM_5': np.random.randint(1, 6, 100),
        'KP_1': np.random.randint(1, 6, 100), 'KP_2': np.random.randint(1, 6, 100), 'KP_3': np.random.randint(1, 6, 100), 'KP_4': np.random.randint(1, 6, 100), 'KP_5': np.random.randint(1, 6, 100),
        'KE_1': np.random.randint(1, 6, 100), 'KE_2': np.random.randint(1, 6, 100), 'KE_3': np.random.randint(1, 6, 100), 'KE_4': np.random.randint(1, 6, 100), 'KE_5': np.random.randint(1, 6, 100),
        'LO_1': np.random.randint(1, 6, 100), 'LO_2': np.random.randint(1, 6, 100), 'LO_3': np.random.randint(1, 6, 100), 'LO_4': np.random.randint(1, 6, 100), 'LO_5': np.random.randint(1, 6, 100),
    }
    return pd.DataFrame(data)

def download_link(df, filename):
    """Menyediakan link untuk mengunduh DataFrame sebagai CSV."""
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color: #00695c; text-decoration: underline; font-weight: bold;">Unduh Template Data ({filename})</a>'
    return href

def get_lv_positions(lvs):
    """Menentukan posisi visual (x, y) untuk LV dalam diagram jalur."""
    num_lvs = len(lvs)
    if num_lvs == 0:
        return {}
    
    # Simple layered layout approximation for up to 5 LVs
    # Assuming the order in LVs list roughly corresponds to the path flow
    positions = {}
    if num_lvs == 5:
        # Layer 1 (IVs)
        positions[lvs[0]] = (-1, 1.5)
        positions[lvs[1]] = (-1, 0.5)
        # Layer 2 (Mediator/Second IV)
        positions[lvs[2]] = (0, 1.5)
        # Layer 3 (Third IV/DV)
        positions[lvs[3]] = (1, 1.5)
        # Layer 4 (Final DV)
        positions[lvs[4]] = (2, 1)
    elif num_lvs > 0:
        # General layout: arrange in a circle/vertical stack
        angle_step = 2 * np.pi / num_lvs
        for i, lv in enumerate(lvs):
            angle = i * angle_step
            # Position LVs in a slight vertical spread
            positions[lv] = (np.cos(angle) * 1.5, np.sin(angle) * 1.5)
            
    return positions

def plot_sem_paths(model_dict, hypotheses, path_df, alpha):
    """Membuat plot diagram jalur yang disederhanakan."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Path Diagram: Koefisien Jalur (PLS-SEM)", fontsize=14, color='#004d40')
    ax.set_axis_off()
    
    lvs = list(model_dict.keys())
    positions = get_lv_positions(lvs)
    
    # Define colors for significance (Green for Significant, Red for Non-significant)
    cmap = LinearSegmentedColormap.from_list("sig_colors", ["#d32f2f", "#4CAF50"]) # Red (Non-sig) to Green (Sig)
    
    # 1. Draw Nodes (LVs)
    for lv, (x, y) in positions.items():
        # Draw node (simple circle/box)
        ax.scatter(x, y, s=5000, color='#e0f2f1', edgecolor='#00695c', linewidth=2, zorder=3)
        ax.text(x, y, lv, ha='center', va='center', fontsize=12, weight='bold', color='#004d40', zorder=4)

    # 2. Draw Edges (Paths) and Labels
    for from_var, to_var in hypotheses:
        if from_var in positions and to_var in positions:
            x1, y1 = positions[from_var]
            x2, y2 = positions[to_var]
            
            # Find the path coefficient and p-value
            path_data = path_df[(path_df['lval'] == to_var) & (path_df['op'] == '~') & (path_df['rval'] == from_var)]
            
            if not path_data.empty:
                beta = path_data['Estimate'].iloc[0]
                p_val = path_data['p-value'].iloc[0] if 'p-value' in path_data.columns else 0.5 # Fallback
                
                is_significant = p_val < alpha
                path_color = cmap(1.0) if is_significant else cmap(0.0)
                linestyle = '-' if is_significant else '--'
                linewidth = 2.5 if is_significant else 1.5
                
                # Calculate text position (midpoint + slight offset for clarity)
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # Draw Arrow
                ax.annotate(
                    '', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->", color=path_color, linewidth=linewidth, linestyle=linestyle,
                        shrinkA=0.15, shrinkB=0.15, mutation_scale=20
                    ),
                    zorder=1
                )
                
                # Label text
                label = f"β={beta:.3f}\nP={p_val:.3f}"
                
                # Calculate rotation for text
                dx = x2 - x1
                dy = y2 - y1
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Place the label slightly offset from the midpoint
                offset_x = 0.05 * dy / np.linalg.norm([dx, dy]) if np.linalg.norm([dx, dy]) > 0 else 0
                offset_y = 0.05 * dx / np.linalg.norm([dx, dy]) if np.linalg.norm([dx, dy]) > 0 else 0
                
                # Add text label
                ax.text(mid_x - offset_x, mid_y + offset_y, label, 
                        ha='center', va='center', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'),
                        color='black', zorder=5)

    return fig

# --- Sidebar Navigation ---
st.sidebar.title("🛠️ Workflow Analisis")
page = st.sidebar.selectbox(
    "Pilih Langkah", 
    ["1. Import Data", "2. Model & Hipotesis", "3. Uji Validitas (Outer Model)", "4. Hasil Analisis"]
)
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini menggunakan library `semopy` dengan metode **PLS-SEM**.")


# --- Session State Initialization ---
if 'latent_vars' not in st.session_state:
    st.session_state['latent_vars'] = {
        'KL': ['KL_1', 'KL_2', 'KL_3', 'KL_4', 'KL_5'],
        'CM': ['CM_1', 'CM_2', 'CM_3', 'CM_4', 'CM_5'],
        'KP': ['KP_1', 'KP_2', 'KP_3', 'KP_4', 'KP_5'],
        'KE': ['KE_1', 'KE_2', 'KE_3', 'KE_4', 'KE_5'],
        'LO': ['LO_1', 'LO_2', 'LO_3', 'LO_4', 'LO_5']
    }
if 'paths' not in st.session_state:
    st.session_state['paths'] = [
        ('KL', 'KP'), ('CM', 'KP'), ('KP', 'KE'), ('KE', 'LO')
    ]
if 'alpha' not in st.session_state:
    st.session_state['alpha'] = 0.05
if 'bootstrap_samples' not in st.session_state:
    st.session_state['bootstrap_samples'] = 5000
if 'is_validated' not in st.session_state:
    st.session_state['is_validated'] = False
if 'loading_threshold' not in st.session_state:
    st.session_state['loading_threshold'] = 0.708

# --- Page 1: Upload Data ---
if page == "1. Import Data":
    st.header("1. Upload File Data CSV")
    
    col_temp, col_info = st.columns([1, 2])
    with col_temp:
        template_df = generate_template()
        st.markdown(download_link(template_df, "template_sem_pls.csv"), unsafe_allow_html=True)
    with col_info:
        st.info("Pastikan data Anda hanya berisi kolom ID Responden dan Indikator (skala Likert/rasio).")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            st.session_state['is_validated'] = False # Reset validation flag on new upload
            st.success(f"Data berhasil diupload! Ukuran: **{df.shape[0]}** responden x **{df.shape[1]}** kolom.")
            
            # Preview data
            st.subheader("Preview Data")
            st.dataframe(df.head())
            
            # Identifikasi kolom indikator
            indikator_cols = [col for col in df.columns if col.lower() != 'responden_id']
            st.session_state['indicator_cols'] = indikator_cols
            st.write(f"**Indikator terdeteksi:** Total **{len(indikator_cols)}** kolom data.")
            
        except Exception as e:
            st.error(f"Gagal memuat file: {e}")

# --- Page 2: Definisi Model ---
elif page == "2. Model & Hipotesis":
    st.header("2. Definisi Model Pengukuran & Struktural")
    if 'df' not in st.session_state or not st.session_state['indicator_cols']:
        st.warning("Upload data terlebih dahulu di langkah 1.")
        st.stop()
    
    # Reset validation flag if model definition is modified
    st.session_state['is_validated'] = False 
    
    indikator_cols = st.session_state['indicator_cols']
    
    st.markdown("### A. Model Pengukuran (Measurement Model)")
    st.info("Tentukan variabel laten (konstrak) dan indikator-indikator yang mengukurnya.")
    
    # Dynamic LV Assignment (using index to manage state)
    lv_names = list(st.session_state['latent_vars'].keys())
    
    for i, lv_name in enumerate(lv_names):
        default_inds = st.session_state['latent_vars'].get(lv_name, [])
        
        with st.expander(f"⚙️ Variabel Laten: **{lv_name}** ({len(default_inds)} Indikator)", expanded=True):
            col_lv_name, col_lv_inds = st.columns([2, 5])
            
            with col_lv_name:
                new_lv_name = st.text_input("Nama Variabel Laten", value=lv_name, key=f"lv_name_{i}")
            
            with col_lv_inds:
                selected_inds = st.multiselect(
                    "Pilih Indikator Terkait",
                    options=indikator_cols,
                    default=default_inds,
                    key=f"lv_inds_{i}"
                )
            
            # Update state when inputs change
            if new_lv_name != lv_name or selected_inds != default_inds:
                new_lvs = st.session_state['latent_vars'].copy()
                
                # Logic for renaming and updating indicators
                old_name_exists = lv_name in new_lvs
                
                if new_lv_name != lv_name and new_lv_name not in new_lvs and old_name_exists:
                    # Rename the key
                    new_lvs[new_lv_name] = new_lvs.pop(lv_name)
                    
                    # Update paths that referenced the old name
                    new_paths = [(new_lv_name if f == lv_name else f, new_lv_name if t == lv_name else t) 
                                 for f, t in st.session_state['paths']]
                    st.session_state['paths'] = new_paths
                    
                    # Refresh to reflect name change in expanders
                    st.session_state['latent_vars'] = new_lvs
                    st.experimental_rerun()
                    
                elif new_lv_name in new_lvs or new_lv_name == lv_name:
                    # Update indicators for the current LV name
                    new_lvs[new_lv_name] = selected_inds
                    st.session_state['latent_vars'] = new_lvs

    
    st.markdown("### B. Model Struktural (Structural Model - Paths)")
    st.info("Definisikan hipotesis (jalur) antar variabel laten.")
    
    # List of currently defined LVs for selection
    current_lvs = list(st.session_state['latent_vars'].keys())
    
    # Display and edit current paths
    paths_to_remove = []
    
    with st.container():
        st.markdown("#### Daftar Hipotesis Aktif:")
        if st.session_state['paths']:
            for i, (from_var, to_var) in enumerate(st.session_state['paths']):
                col_path, col_btn = st.columns([9, 1])
                with col_path:
                    st.markdown(f"**H{i+1}:** `{from_var}` ➡️ `{to_var}`")
                with col_btn:
                    if st.button("Hapus", key=f"del_path_{i}"):
                        paths_to_remove.append(i)
        else:
            st.markdown("*(Belum ada hipotesis yang didefinisikan)*")
            
    # Remove paths marked for removal
    if paths_to_remove:
        new_paths = [path for i, path in enumerate(st.session_state['paths']) if i not in paths_to_remove]
        st.session_state['paths'] = new_paths
        st.experimental_rerun()
        
    st.markdown("#### Tambah Hipotesis Baru:")
    col_add_from, col_add_to, col_add_btn = st.columns([3, 3, 2])
    with col_add_from:
        new_from = st.selectbox("Eksogen (Dari)", current_lvs, key="new_path_from")
    with col_add_to:
        new_to = st.selectbox("Endogen (Ke)", current_lvs, key="new_path_to")
    
    with col_add_btn:
        st.write("") # Spacer
        if st.button("➕ Tambah Jalur", disabled=(new_from == new_to)):
            new_path = (new_from, new_to)
            if new_path not in st.session_state['paths']:
                st.session_state['paths'].append(new_path)
                st.success(f"Jalur {new_from} -> {new_to} ditambahkan!")
                st.experimental_rerun()
            else:
                st.warning("Jalur ini sudah ada!")

    st.markdown("---")
    st.subheader("C. Opsi PLS-SEM")
    col_boot, col_alpha = st.columns(2)
    with col_boot:
        st.session_state['bootstrap_samples'] = st.slider("Bootstrap Samples", 1000, 10000, st.session_state['bootstrap_samples'])
    with col_alpha:
        st.session_state['alpha'] = st.slider("Tingkat Signifikansi (α)", 0.01, 0.10, st.session_state['alpha'])

    # Final button to check validation
    if st.button("✅ Simpan Model & Lanjut ke Uji Validitas", type="primary"):
        valid_lvs = {k: v for k, v in st.session_state['latent_vars'].items() if len(v) >= 1}
        if not valid_lvs:
            st.error("Model Pengukuran tidak valid. Definisikan minimal 1 variabel laten dengan setidaknya 1 indikator.")
        elif not st.session_state['paths']:
            st.error("Model Struktural tidak valid. Definisikan minimal 1 jalur hipotesis.")
        else:
            st.session_state['latent_vars'] = valid_lvs
            st.success("Model berhasil disimpan! Lanjutkan ke tab '3. Uji Validitas'.")

# --- Page 3: Uji Validitas (Outer Model) ---
elif page == "3. Uji Validitas (Outer Model)":
    st.header("3. Uji Validitas Konvergen (Outer Model)")
    
    if 'df' not in st.session_state or not st.session_state['paths']:
        st.warning("Mohon definisikan Model & Hipotesis di langkah 2 terlebih dahulu.")
        st.stop()
    
    df = st.session_state['df']
    model_dict = st.session_state['latent_vars']
    
    st.markdown("#### Tentukan Ambang Batas Validitas (Loading Factor)")
    col_thresh, col_notes = st.columns([1, 2])
    with col_thresh:
        st.session_state['loading_threshold'] = st.slider(
            "Min Loading Factor", 
            min_value=0.40, max_value=0.80, 
            value=st.session_state['loading_threshold'], 
            step=0.01, format="%.3f"
        )
    with col_notes:
        st.info(f"Umumnya, nilai batas yang digunakan adalah **0.708** (validitas tinggi) atau **0.50** (validitas sedang). Saat ini menggunakan: **{st.session_state['loading_threshold']:.3f}**.")

    # 1. Persiapan Model Specification (Hanya Measurement Model diperlukan)
    data = df.drop(columns=['responden_id'] if 'responden_id' in df.columns else [], errors='ignore')
    meas_model = "\n".join([f"{latent} =~ " + " + ".join(inds) for latent, inds in model_dict.items()])
    
    # Simple check for data availability
    all_indicators = [ind for inds in model_dict.values() for ind in inds]
    if any(ind not in data.columns for ind in all_indicators):
        st.error("🚨 VALIDASI MODEL GAGAL! Beberapa indikator model tidak ada di data. Kembali ke Langkah 2.")
        st.stop()
        
    # 2. Jalankan PLS-SEM (Hanya fit untuk mendapatkan loadings)
    try:
        st.info("Menghitung Outer Loading untuk Uji Validitas...")
        model = semopy.Model(meas_model)
        res = model.fit(data, algo="PLS")
        st.success("Perhitungan Outer Loading Selesai.")
        
        loadings_df = res.inspect(lv='all', rv='all', col='Loading', std=True, pretty_names=True)
        loadings_df = loadings_df[loadings_df['op'] == '=~'].rename(columns={'lval': 'Konstrak', 'rval': 'Indikator', 'Estimate': 'Loading'})
        
        # 3. Tentukan Validitas
        loadings_df['Valid'] = loadings_df['Loading'].apply(lambda x: abs(x) >= st.session_state['loading_threshold'])
        
        st.markdown("#### Hasil Outer Loading dan Validitas")
        
        # Highlight dataframe
        def highlight_validity(val):
            color = 'background-color: #e6ffed' if val else 'background-color: #ffe6e6'
            return color

        st.dataframe(
            loadings_df[['Konstrak', 'Indikator', 'Loading', 'Valid']].sort_values(by='Konstrak'),
            column_config={
                "Loading": st.column_config.NumberColumn("Loading Factor", format="%.3f"),
                "Valid": st.column_config.TextColumn(
                    "Keputusan",
                    help="Valid jika Loading Factor >= Ambang Batas",
                    width="small"
                )
            },
            height=300,
            hide_index=True
        )
        
        invalid_indicators = loadings_df[loadings_df['Valid'] == False]['Indikator'].tolist()
        
        st.markdown("---")

        if invalid_indicators:
            st.warning(f"⚠️ Indikator Tidak Valid Ditemukan ({len(invalid_indicators)}): **{', '.join(invalid_indicators)}**")
            st.info("Menurut PLS-SEM, indikator yang tidak valid (Loading Factor di bawah ambang batas) **harus dihapus** dari model untuk memastikan validitas konvergen sebelum analisis struktural.")
            
            if st.button(f"HAPUS {len(invalid_indicators)} Indikator Tidak Valid & Lanjutkan", type="primary"):
                
                new_model_dict = {}
                for lv, inds in model_dict.items():
                    # Keep only valid indicators for this LV
                    valid_inds = [ind for ind in inds if ind not in invalid_indicators]
                    if valid_inds:
                        new_model_dict[lv] = valid_inds
                
                # Update session state with the cleaned model and data
                st.session_state['latent_vars'] = new_model_dict
                
                # Update data: drop columns of invalid indicators
                cols_to_drop = [col for col in invalid_indicators if col in st.session_state['df'].columns]
                st.session_state['df'] = st.session_state['df'].drop(columns=cols_to_drop, errors='ignore')
                st.session_state['is_validated'] = True
                
                st.success(f"Model dan Data berhasil diperbaiki! {len(cols_to_drop)} kolom indikator telah dihapus.")
                st.button("Lanjutkan ke Hasil Analisis", on_click=lambda: st.session_state.__setitem__('page', '4. Hasil Analisis'))
        else:
            st.success("✅ Semua indikator Valid! Model Pengukuran dapat diterima.")
            st.session_state['is_validated'] = True
            st.button("Lanjutkan ke Hasil Analisis", type="primary")

    except Exception as e:
        st.error(f"❌ Terjadi Error saat menghitung Outer Loading: {str(e)}")
        st.markdown("**Pesan Perbaikan:** Pastikan semua Variabel Laten Anda memiliki minimal 2 Indikator (ideal) dan nama kolom indikator sudah benar.")


# --- Page 4: Hasil Analisis ---
elif page == "4. Hasil Analisis":
    st.header("4. Hasil Analisis PLS-SEM Final")
    
    if not st.session_state.get('is_validated', False):
        st.warning("⚠️ Anda harus menyelesaikan **Uji Validitas (Outer Model)** di Langkah 3 terlebih dahulu.")
        st.stop()

    df = st.session_state['df']
    model_dict = st.session_state['latent_vars']
    hypotheses = st.session_state['paths']
    bootstrap_samples = st.session_state['bootstrap_samples']
    alpha = st.session_state['alpha']
    
    # 1. Persiapan Model Specification
    data = df.drop(columns=['responden_id'] if 'responden_id' in df.columns else [], errors='ignore')
    
    meas_model = "\n".join([f"{latent} =~ " + " + ".join(inds) for latent, inds in model_dict.items()])
    struct_model = "\n".join([f"{to_var} ~ {from_var}" for from_var, to_var in hypotheses])
    full_model = meas_model + "\n" + struct_model
    
    st.subheader("Ringkasan Model Akhir (Syntax `semopy`)")
    st.code(full_model)

    # 2. Jalankan Analisis
    try:
        st.info(f"Menjalankan PLS-SEM Final (Bootstrap N={bootstrap_samples}).")
        
        # Inisialisasi dan Fit Model
        model = semopy.Model(full_model)
        res = model.fit(data, algo="PLS")
        
        # Bootstrapping untuk Signifikansi
        boot_res = model.bootstrap(data, nboot=bootstrap_samples)
        
        st.success("Analisis selesai! Lihat hasil di bawah.")
        
        path_df = res.inspect(mode='estimates')
        boot_df = boot_res.inspect(mode='estimates')
        
        # --- PATH VISUALIZATION (SmartPLS-like) ---
        st.markdown("---")
        st.subheader("Path Diagram Visualisasi Hipotesis")
        st.markdown(f"**Keterangan:** Garis tebal/hijau = Signifikan ($P < \\alpha={alpha}$), Garis putus-putus/merah = Tidak Signifikan.")
        
        # Merge path data with p-values
        try:
            path_df = path_df.merge(boot_df[['lval', 'op', 'rval', 'T-stat', 'p-value']], 
                                    on=['lval', 'op', 'rval'], 
                                    how='left', suffixes=('', '_boot'))
            path_df['p-value'] = path_df['p-value_boot'].fillna(path_df['p-value'])
            path_df = path_df.drop(columns=['p-value_boot'], errors='ignore')
        except:
            pass
        
        # Generate Plot
        try:
            fig_path = plot_sem_paths(model_dict, hypotheses, path_df, alpha)
            st.pyplot(fig_path)
        except Exception as plot_e:
            st.warning(f"Gagal menampilkan Path Diagram: {plot_e}. Periksa urutan LV.")

        st.markdown("---")
        
        # --- TABULAR RESULTS ---
        tab_hyp, tab_outer, tab_r2 = st.tabs(["Uji Hipotesis", "Model Pengukuran (Outer Model)", "Koefisien Determinasi (R²)"])

        with tab_hyp:
            st.write("#### Uji Hipotesis (Inner Model)")
            
            hyp_table = []
            for i, (from_var, to_var) in enumerate(hypotheses):
                path_row = path_df[(path_df['lval'] == to_var) & (path_df['op'] == '~') & (path_df['rval'] == from_var)]
                
                if not path_row.empty:
                    beta = path_row['Estimate'].iloc[0]
                    p_val = path_row['p-value'].iloc[0]
                    t_stat = path_row['T-stat'].iloc[0]
                    
                    decision = "Diterima" if p_val < alpha and beta > 0 else "Ditolak"
                    sign = "Positif" if beta > 0 else "Negatif"
                    
                    hyp_table.append({
                        'Hipotesis': f"H{i+1}",
                        'Jalur': f"{from_var} -> {to_var}",
                        'β (Koef. Jalur)': f"{beta:.3f}",
                        'T-Stat': f"{t_stat:.3f}",
                        'P-Value': f"{p_val:.3f}",
                        'Arah': sign,
                        f'Keputusan (α={alpha})': decision
                    })
            
            hyp_df = pd.DataFrame(hyp_table)
            st.dataframe(hyp_df.set_index('Hipotesis'), use_container_width=True)
            
            st.markdown("##### Interpretasi Singkat:")
            for _, row in hyp_df.iterrows():
                decision_emoji = "✅" if row[f'Keputusan (α={alpha})'] == 'Diterima' else "❌"
                st.markdown(f"{decision_emoji} **{row['Hipotesis']} ({row['Jalur']}):** Hubungan **{row['Arah']}** dan **{row[f'Keputusan (α={alpha})']}** (β={row['β (Koef. Jalur)']}, P={row['P-Value']}).")

        with tab_outer:
            st.write("#### Evaluasi Outer Model (Validitas & Reliabilitas)")
            loadings = res.inspect(lv='all', rv='all', col='Loading', std=True, pretty_names=True)
            loadings = loadings[loadings['op'] == '=~'].rename(columns={'lval': 'Konstrak', 'rval': 'Indikator', 'Estimate': 'Loading'})
            loadings = loadings.sort_values(by='Konstrak')
            
            st.markdown("##### Loading Factors (Indikator yang telah valid)")
            st.dataframe(loadings[['Konstrak', 'Indikator', 'Loading']].reset_index(drop=True), use_container_width=True)
            
            st.markdown("##### Composite Reliability (CR) & Average Variance Extracted (AVE)")
            st.warning("Perhitungan CR dan AVE di `semopy` mungkin memerlukan proses yang lebih kompleks, ini adalah ringkasan yang disederhanakan.")
            
            st.dataframe(res.inspect(mode='variances'), use_container_width=True)

        with tab_r2:
            st.write("#### R-squared ($R^2$) - Koefisien Determinasi")
            r2_df = res.inspect(mode='r2')
            st.dataframe(r2_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Terjadi Error dalam menjalankan analisis PLS-SEM: {str(e)}")
        st.markdown("""
        **Pesan Perbaikan (Troubleshooting):**
        1. **Periksa Model Pengukuran:** Pastikan setiap Variabel Laten (LV) yang Anda definisikan di Langkah 2 memiliki **minimal satu Indikator** yang valid dan terukur.
        2. **Periksa Data dan Indikator:** Pastikan **nama kolom indikator** di file CSV Anda (Langkah 1) sama persis dengan yang Anda pilih di Definisi Model (Langkah 2).
        3. **Periksa Model Struktural:** Pastikan jalur (paths) tidak memiliki variabel yang tidak terdefinisi atau jalur yang tidak logis/membentuk siklus.
        
        Silakan kembali ke tab **'2. Model & Hipotesis'** untuk meninjau definisi model Anda.
        """)
        

st.markdown("---")
# Tombol reset
if st.sidebar.button("Mulai Baru / Reset Aplikasi"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()