import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from datetime import datetime
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

# ==============================================================================
# 1. CONFIGURACIÓN DEL UNIVERSO (SETUP)
# ==============================================================================
st.set_page_config(
    page_title="GEOESTADISTICA MINERA - KRIGING PRO", 
    page_icon="⚒️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. PROTOCOLO VISUAL (ESTILOS CSS AVANZADOS & DARK MODE)
# ==============================================================================
st.markdown("""
    <style>
    /* --- CONFIGURACIÓN GENERAL --- */
    .stApp {
        background-color: #0e1117; 
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* --- TIPOGRAFÍA --- */
    h1, h2, h3 {color: #ffffff; font-weight: 600;}
    h4, h5, h6 {color: #b0bec5;}
    p, li, label, span {font-size: 16px; line-height: 1.6;}
    
    /* --- CAJAS EDUCATIVAS (TEORÍA) --- */
    .theory-box {
        background-color: #1a2332; 
        border-left: 6px solid #00bcd4;
        padding: 25px; 
        border-radius: 10px; 
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); 
        color: #e1f5fe;
    }
    .theory-title {
        color: #4dd0e1; 
        font-weight: 800; 
        font-size: 1.3em; 
        display: block; 
        margin-bottom: 12px;
        text-transform: uppercase; 
        letter-spacing: 1.2px;
        border-bottom: 1px solid #4dd0e1;
        padding-bottom: 5px;
    }
    
    /* --- CAJAS DE RESULTADOS (EXITO) --- */
    .result-box {
        background-color: #1b3a25;
        border-left: 6px solid #00e676;
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px;
        color: #e8f5e9;
        box-shadow: 0 4px 10px rgba(0,230,118,0.1);
    }

    /* --- CAJAS DE MATEMÁTICA (EXPLICACIÓN DE FÓRMULAS) --- */
    .math-step {
        background-color: #263238;
        border: 1px solid #37474f;
        border-left: 4px solid #ffca28;
        padding: 15px; 
        border-radius: 5px; 
        margin-top: 15px;
        color: #eceff1; 
        font-family: 'Courier New', monospace;
        font-size: 0.95em;
    }

    /* --- SEMÁFORO JORC (CLASIFICACIÓN) --- */
    .jorc-card {
        padding: 25px; 
        border-radius: 15px; 
        text-align: center; 
        margin-bottom: 20px;
        transition: transform 0.3s;
    }
    .jorc-card:hover { transform: scale(1.02); }
    
    .jorc-medido {
        background: linear-gradient(145deg, #1b5e20, #2e7d32); 
        border: 2px solid #66bb6a; 
        color: #ffffff;
        box-shadow: 0 0 20px rgba(102, 187, 106, 0.4);
    }
    .jorc-indicado {
        background: linear-gradient(145deg, #e65100, #f57c00); 
        border: 2px solid #ffb74d; 
        color: #ffffff;
        box-shadow: 0 0 20px rgba(255, 183, 77, 0.4);
    }
    .jorc-inferido {
        background: linear-gradient(145deg, #b71c1c, #c62828); 
        border: 2px solid #ef5350; 
        color: #ffffff;
        box-shadow: 0 0 20px rgba(239, 83, 80, 0.4);
    }

    /* --- BOTONES PERSONALIZADOS --- */
    .stButton>button {
        background: linear-gradient(90deg, #0277bd 0%, #01579b 100%); 
        color: white; 
        border: none;
        border-radius: 8px; 
        height: 55px; 
        font-weight: bold; 
        font-size: 1.2em;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0288d1 0%, #0277bd 100%);
        box-shadow: 0 8px 15px rgba(2, 119, 189, 0.6);
        transform: translateY(-2px);
    }

    /* --- PESTAÑAS (TABS) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; 
        background-color: #0e1117;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f2937; 
        color: #b0bec5; 
        border: 1px solid #374151; 
        border-radius: 5px 5px 0 0;
        padding: 12px 25px;
        font-size: 1.1em;
    }
    .stTabs [aria-selected="true"] {
        background-color: #263238; 
        border-top: 4px solid #00bcd4; 
        color: #ffffff;
        font-weight: bold;
    }
    
    /* --- TABLAS DATAFRAME --- */
    [data-testid="stDataFrame"] {
        border: 1px solid #374151;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. GESTIÓN DE ESTADO Y DATOS (BACKEND)
# ==============================================================================

# Inicialización de Datos (Ejemplo PDF "Proyecto Altiplano Sur")
if 'df_data' not in st.session_state:
    st.session_state['df_data'] = pd.DataFrame({
        'ID': ['DDH-101', 'DDH-102', 'DDH-103', 'DDH-104'],
        'X': [385250.0, 385275.0, 385300.0, 385320.0],
        'Y': [8245100.0, 8245125.0, 8245080.0, 8245150.0],
        'Ley': [0.85, 1.12, 0.72, 0.95]
    })

if 'resultado' not in st.session_state:
    st.session_state['resultado'] = None

# Función: Cargar Archivo CSV (Blindada contra errores de tipo)
def cargar_csv():
    """Carga y valida el archivo CSV subido por el usuario."""
    uploaded = st.session_state.uploader_key
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            # Normalizar nombres de columnas
            cols = [c.strip().capitalize() for c in df.columns]
            df.columns = cols
            
            # Validación de columnas críticas
            required_cols = {'X', 'Y', 'Ley'}
            if required_cols.issubset(df.columns):
                # Generar IDs si no existen
                if 'Id' not in df.columns: 
                    df['Id'] = [f"M-{i+1}" for i in range(len(df))]
                # Convertir a string para evitar errores de formato posteriores
                df['Id'] = df['Id'].astype(str)
                st.session_state['df_data'] = df
                st.toast("✅ Base de datos actualizada correctamente.")
            else:
                st.error(f"❌ El archivo debe contener las columnas: {required_cols}")
        except Exception as e:
            st.error(f"Error crítico al leer el archivo: {e}")

# Función: Guardar Historial (Persistencia)
def guardar_historial(res):
    """Guarda cada cálculo realizado en un archivo CSV local."""
    archivo = 'historial_proyecto.csv'
    nuevo_registro = {
        'Fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'X_Bloque': res['tx'], 
        'Y_Bloque': res['ty'],
        'Ley_Estimada': round(res['ley'], 4),
        'Varianza_Kriging': round(res['var'], 4),
        'CV_Kriging': round(res['cv_k'], 2),
        'Categoria_JORC': res['cat']
    }
    df_new = pd.DataFrame([nuevo_registro])
    
    # Modo 'append' si existe, modo 'write' si no
    if not os.path.exists(archivo):
        df_new.to_csv(archivo, index=False)
    else:
        df_new.to_csv(archivo, mode='a', header=False, index=False)

# ==============================================================================
# 4. MOTOR MATEMÁTICO (CORE DE CÁLCULO)
# ==============================================================================

def variograma_esferico(h, c0, c1, a):
    """Calcula el valor del variograma teórico usando el modelo Esférico."""
    h = np.atleast_1d(h)
    val = np.zeros_like(h)
    c_total = c0 + c1 
    
    mask_r = h > a
    val[mask_r] = c_total
    
    mask_i = ~mask_r
    val[mask_i] = c0 + c1 * (1.5 * (h[mask_i] / a) - 0.5 * (h[mask_i] / a)**3)
    val[h == 0] = 0
    return val

def resolver_kriging(df, target, c0, c1, a):
    """Resuelve el sistema de ecuaciones de Kriging Ordinario (OK)."""
    try:
        coords = df[['X', 'Y']].values
        leyes = df['Ley'].values
        n = len(coords)
        
        # 1. Distancias
        dist_mat = cdist(coords, coords)
        dist_target = cdist(coords, [target]).flatten()
        
        # 2. Matrices Kriging
        K = np.zeros((n+1, n+1))
        M = np.zeros(n+1)
        
        K_vals = variograma_esferico(dist_mat, c0, c1, a)
        K[:n, :n] = K_vals
        K[n, :] = 1.0; K[:, n] = 1.0; K[n, n] = 0.0
        
        M_vals = variograma_esferico(dist_target, c0, c1, a)
        M[:n] = M_vals; M[n] = 1.0 
        
        # 3. Resolución
        W = np.linalg.solve(K, M)
        pesos = W[:n]
        mu = W[n]
        
        # 4. Resultados
        ley_est = np.sum(pesos * leyes)
        var_krig = np.sum(pesos * M_vals) + mu
        sigma_k = np.sqrt(var_krig) if var_krig > 0 else 0
        
        # 5. JORC
        cv_kriging = (sigma_k / ley_est * 100) if ley_est > 0 else 100
        var_global = np.var(leyes, ddof=1) if len(leyes) > 1 else 1.0
        slope = 1.0 - (var_krig / var_global) if var_global > 0 else 0
        
        if cv_kriging < 15: cat = "MEDIDO"
        elif 15 <= cv_kriging <= 30: cat = "INDICADO"
        else: cat = "INFERIDO"
        
        return {
            'status': 'OK', 
            'ley': ley_est, 'var': var_krig, 'sigma': sigma_k,
            'cv_k': cv_kriging, 'slope': slope, 'cat': cat,
            'pesos': pesos, 'mu': mu, 'K': K, 'M': M, 
            'd_mat': dist_mat, 'd_vec': dist_target
        }
    except Exception as e:
        return {'status': 'ERROR', 'msg': str(e)}

# ==============================================================================
# 5. INTERFAZ GRÁFICA (FRONTEND)
# ==============================================================================

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
    st.markdown("## 🏗️ PANEL DE CONTROL")
    st.markdown("---")
    st.markdown("**📁 Proyecto:** POMPERIA S.A.C.")
    st.markdown("**👷 Estudiante:** Felix Bautista Layme")
    st.markdown("**📅 Fecha:** " + datetime.now().strftime("%d/%m/%Y"))
    
    st.markdown("---")
    st.markdown("### 📚 GLOSARIO MINERO")
    with st.expander("Ver Definiciones"):
        st.markdown("""
        **1. Kriging Ordinario:** Método geoestadístico de estimación que minimiza la varianza del error y es insesgado.
        **2. Nugget (C0):** Variabilidad a muy corta distancia.
        **3. Sill (C):** Varianza total de los datos.
        **4. Rango (a):** Distancia hasta donde las muestras tienen correlación espacial.
        """)
    
    st.success("💾 **Sistema Guardando Automáticamente**")

st.title("GEOESTADÍSTICA MINERA - KRIGING")
st.markdown("#### Simulador Profesional de Estimación de Recursos Minerales (v28.0)")

tabs = st.tabs([
    "📊 1. Análisis de Datos", 
    "📈 2. Variografía", 
    "⚙️ 3. Estimación", 
    "🧮 4. Ingeniería Inversa (Detallada)",
    "💰 5. Economía",
    "⚖️ 6. JORC (Avanzado)", 
    "📜 7. Informe Final"
])

# --- TAB 1: ANÁLISIS DE DATOS ---
with tabs[0]:
    st.markdown("""
    <div class='theory-box'>
        <span class='theory-title'>🔍 Módulo 1: Validación de Datos (QA/QC)</span>
        El primer paso de cualquier estimación es conocer la muestra. Aquí buscamos <b>comportamientos anómalos</b> que puedan invalidar el Kriging.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("Base de Datos")
        st.file_uploader("Cargar Archivo CSV", type=['csv'], key="uploader_key", on_change=cargar_csv)
        st.data_editor(st.session_state['df_data'], num_rows="dynamic", height=250)
        st.info("💡 Tip: Verifica que no haya leyes negativas o coordenadas cero.")

    with c2:
        st.subheader("Estadística Descriptiva")
        df = st.session_state['df_data']
        
        # --- LIMPIEZA DE DATOS SEGURA ---
        df_calc = df.copy()
        cols_numericas = ['X', 'Y', 'Ley']
        for col in cols_numericas:
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')
        df_calc = df_calc.dropna(subset=cols_numericas)
        
        if not df_calc.empty:
            media = df_calc['Ley'].mean()
            std = df_calc['Ley'].std()
            cv = (std/media*100) if media>0 else 0
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Ley Media", f"{media:.2f} %")
            k2.metric("Desviación Std.", f"{std:.2f}")
            k3.metric("Coef. Variación (CV)", f"{cv:.1f} %", delta="¡Atención!" if cv>100 else "Ok", delta_color="inverse")
            
            st.markdown(f"""
            <div class='explain-box'>
                <b>📊 Interpretación del C.V. ({cv:.1f}%):</b><br>
                <ul>
                    <li><b>CV < 50%:</b> Datos muy homogéneos. Ideal para Kriging Lineal.</li>
                    <li><b>50% < CV < 100%:</b> Variabilidad moderada.</li>
                    <li><b>CV > 100%:</b> Datos erráticos. Significa que tienes "pepitas" de alta ley.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            t1, t2 = st.tabs(["Distribución", "Análisis de Deriva"])
            with t1:
                fig_h = px.histogram(df_calc, x="Ley", nbins=10, title="Histograma de Frecuencias", color_discrete_sequence=['#00e676'])
                fig_h.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_h, use_container_width=True)
            
            with t2:
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if len(df_calc) > 1:
                        slope, intercept = np.polyfit(df_calc['X'], df_calc['Ley'], 1)
                        fig_x = px.scatter(df_calc, x='X', y='Ley', title="Tendencia Este-Oeste")
                        fig_x.add_trace(go.Scatter(x=df_calc['X'], y=slope*df_calc['X']+intercept, mode='lines', line=dict(color='red'), name='Tendencia'))
                        fig_x.update_layout(template="plotly_dark", height=300)
                        st.plotly_chart(fig_x, use_container_width=True)
                    else:
                        st.warning("Se necesitan al menos 2 datos.")
                        
                with col_d2:
                    if len(df_calc) > 1:
                        slope_y, intercept_y = np.polyfit(df_calc['Y'], df_calc['Ley'], 1)
                        fig_y = px.scatter(df_calc, x='Y', y='Ley', title="Tendencia Norte-Sur")
                        fig_y.add_trace(go.Scatter(x=df_calc['Y'], y=slope_y*df_calc['Y']+intercept_y, mode='lines', line=dict(color='red'), name='Tendencia'))
                        fig_y.update_layout(template="plotly_dark", height=300)
                        st.plotly_chart(fig_y, use_container_width=True)
                    else:
                        st.warning("Se necesitan al menos 2 datos.")

# --- TAB 2: VARIOGRAFÍA ---
with tabs[1]:
    st.markdown("""
    <div class='theory-box'>
        <span class='theory-title'>🔍 Módulo 2: Modelamiento Estructural</span>
        El variograma define la continuidad espacial. Ajusta la curva azul (teórica) para que represente la geología del yacimiento.
    </div>
    """, unsafe_allow_html=True)
    
    cv1, cv2 = st.columns([1, 2])
    with cv1:
        st.subheader("Parámetros del Modelo")
        v_c0 = st.number_input("C0 (Nugget - Efecto Pepita)", 0.0, 10.0, 0.015, format="%.3f")
        v_c1 = st.number_input("C1 (Meseta Parcial)", 0.0, 50.0, 0.085, format="%.3f")
        st.info(f"**Meseta Total (Sill):** {v_c0+v_c1:.3f}")
        v_a  = st.number_input("Rango (Alcance)", 1.0, 500.0, 120.0, format="%.1f")
        
        st.markdown("""
        <div class='result-box'>
            <b>Definiciones:</b>
            <ul>
                <li><b>Nugget:</b> Error a distancia cero.</li>
                <li><b>Sill:</b> Varianza total.</li>
                <li><b>Rango:</b> Distancia de influencia máxima.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with cv2:
        h = np.linspace(0, v_a * 1.5, 100)
        gamma = variograma_esferico(h, v_c0, v_c1, v_a)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h, y=gamma, mode='lines', name='Modelo Esférico', line=dict(color='#00bcd4', width=4)))
        fig.add_hline(y=v_c0+v_c1, line_dash="dash", annotation_text="Meseta Total")
        fig.add_vline(x=v_a, line_dash="dash", annotation_text="Rango")
        fig.update_layout(title="Curva del Variograma Teórico", xaxis_title="Distancia (h)", yaxis_title="Gamma (γ)", template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: ESTIMACIÓN ---
with tabs[2]:
    st.markdown("""
    <div class='theory-box'>
        <span class='theory-title'>⚙️ Módulo 3: Estimación de Bloque</span>
        El sistema resolverá las ecuaciones matriciales para encontrar los pesos óptimos y estimar la ley del bloque.
    </div>
    """, unsafe_allow_html=True)

    c_izq, c_der = st.columns([1, 2])
    with c_izq:
        st.subheader("Configuración")
        tx = st.number_input("Este (X) Bloque", value=385280.0)
        ty = st.number_input("Norte (Y) Bloque", value=8245105.0)
        st.divider()
        if st.button("🚀 EJECUTAR KRIGING"):
            res = resolver_kriging(df_calc, [tx, ty], v_c0, v_c1, v_a)
            if res['status'] == 'OK':
                res.update({'tx': tx, 'ty': ty, 'c0': v_c0, 'c1': v_c1, 'a': v_a, 'fecha': datetime.now()})
                st.session_state['resultado'] = res
                guardar_historial(res)
                st.success("¡Cálculo Exitoso!")
            else:
                st.error(res['msg'])

    with c_der:
        if st.session_state['resultado'] and st.session_state['resultado']['status']=='OK':
            res = st.session_state['resultado']
            st.markdown(f"""
            <div style="background:#1f2937; border:2px solid #00e676; border-radius:10px; padding:20px; text-align:center;">
                <h3 style="color:#00e676; margin:0;">LEY ESTIMADA (Z*)</h3>
                <h1 style="color:white; font-size:4em; margin:10px 0;">{res['ley']:.4f} %</h1>
                <div style="display:flex; justify-content:space-around; margin-top:15px;">
                    <div>Varianza ($\sigma_k^2$)<br><b style="color:#b3e5fc; font-size:1.2em;">{res['var']:.4f}</b></div>
                    <div>Desv. Std ($\sigma_k$)<br><b style="color:#b3e5fc; font-size:1.2em;">{res['sigma']:.4f}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            fig = px.scatter(df_calc, x='X', y='Y', size='Ley', color='Ley', title="Plano de Estimación 2D", color_continuous_scale='Viridis')
            fig.add_trace(go.Scatter(x=[tx], y=[ty], mode='markers+text', marker=dict(color='red', size=25, symbol='x'), text=["BLOQUE"], textposition="top center"))
            t = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(x=tx+v_a*np.cos(t), y=ty+v_a*np.sin(t), mode='lines', line=dict(dash='dash', color='white'), name='Radio Influencia'))
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: INGENIERÍA INVERSA ---
with tabs[3]:
    if st.session_state['resultado']:
        res = st.session_state['resultado']
        st.markdown("""
        <div class='theory-box'>
            <span class='theory-title'>🧮 Módulo 4: Desglose Matemático (Paso a Paso)</span>
            Aquí abrimos la "caja negra" para que veas la matemática exacta detrás del resultado.
        </div>
        """, unsafe_allow_html=True)
        
        # --- PASO 1 ---
        st.markdown("### Paso 1: Cálculo de Distancias Geométricas")
        ids = [str(x) for x in df_calc.get('Id', [f'M-{i+1}' for i in range(len(df_calc))])]
        data_dist = []
        for i in range(len(df_calc)):
            dist = res['d_vec'][i]
            dx = df_calc['X'].iloc[i] - res['tx']
            dy = df_calc['Y'].iloc[i] - res['ty']
            formula_visible = f"√(({dx:.1f})² + ({dy:.1f})²)"
            data_dist.append([ids[i], df_calc['X'].iloc[i], df_calc['Y'].iloc[i], formula_visible, dist])
        
        df_dist_show = pd.DataFrame(data_dist, columns=["Sondaje", "Este (X)", "Norte (Y)", "Fórmula Aplicada", "Distancia (m)"])
        st.dataframe(df_dist_show.style.format(subset=["Este (X)", "Norte (Y)", "Distancia (m)"], formatter="{:.2f}"))

        # --- PASO 2 ---
        st.markdown("### Paso 2: Cálculo de Valores Gamma")
        data_var = []
        for i in range(len(df_calc)):
            h = res['d_vec'][i]
            gam = res['M'][i]
            data_var.append([ids[i], h, gam])
        st.dataframe(pd.DataFrame(data_var, columns=["Sondaje", "Distancia (h)", "Gamma γ(h)"]).style.format(subset=["Distancia (h)", "Gamma γ(h)"], formatter="{:.4f}"))

        # --- PASO 3 ---
        st.markdown("### Paso 3: Ponderación y Ecuación Final")
        st.markdown("""
        <div class='math-step'>
            <b>¿De dónde salió el resultado?</b> Multiplicamos la <b>Ley Real</b> de cada sondaje por su <b>Peso (λ)</b>.
        </div>
        """, unsafe_allow_html=True)
        
        leyes_reales = df_calc['Ley'].values
        pesos_calc = res['pesos']
        aportes = leyes_reales * pesos_calc
        
        ecuacion_str = ""
        for i in range(len(leyes_reales)):
            simbolo = " + " if i > 0 else ""
            ecuacion_str += f"{simbolo}({leyes_reales[i]:.2f} × {pesos_calc[i]:.4f})"
        
        st.latex(r"Z^* = \sum (\lambda_i \cdot Z_i)")
        st.write(f"**Sustituyendo tus valores:**")
        st.markdown(f"`Z* = {ecuacion_str}`")
        
        df_w = pd.DataFrame({
            'Sondaje': ids,
            'Ley Real (%)': leyes_reales,
            'Peso (λ)': pesos_calc,
            'Aporte (%)': aportes
        })
        st.dataframe(df_w.style.format(subset=['Ley Real (%)', 'Peso (λ)', 'Aporte (%)'], formatter="{:.4f}").background_gradient(subset=['Peso (λ)'], cmap='Greens'))
        
        total_aporte = np.sum(aportes)
        st.markdown(f"### 👉 SUMA TOTAL (Z*) = **{total_aporte:.4f} %**")

        # --- PASO 4 ---
        st.markdown("### Paso 4: Varianza de Kriging")
        suma_prod = np.sum(res['pesos'] * res['M'][:-1])
        st.write(f"• Suma (Pesos × Gammas): **{suma_prod:.5f}**")
        st.write(f"• Multiplicador Lagrange (μ): **{res['mu']:.5f}**")
        st.success(f"**Varianza Total:** {suma_prod:.5f} + {res['mu']:.5f} = **{res['var']:.4f}**")

# ==============================================================================
# TAB 5: ECONOMÍA
# ==============================================================================
with tabs[4]:
    if st.session_state['resultado']:
        res = st.session_state['resultado']
        st.markdown("""
        <div class='theory-box'>
            <span class='theory-title'>💰 Módulo 5: Valorización Económica</span>
            Transformamos la Ley Geológica (%) en Valor Económico (US$).
        </div>
        """, unsafe_allow_html=True)
        
        ce1, ce2 = st.columns(2)
        with ce1:
            st.markdown("**Parámetros Económicos:**")
            dim_x = st.number_input("Largo Bloque (m)", value=25.0)
            dim_y = st.number_input("Ancho Bloque (m)", value=25.0)
            dim_z = st.number_input("Alto Bloque (m)", value=15.0)
            densidad = st.number_input("Densidad (t/m³)", value=2.65)
            precio = st.number_input("Precio Cobre (US$/lb)", value=3.85)
            recup = st.number_input("Recuperación (%)", value=85.0)
            
        with ce2:
            volumen = dim_x * dim_y * dim_z
            tonelaje = volumen * densidad
            fino_ton = tonelaje * (res['ley']/100)
            fino_lb = fino_ton * 2204.62
            fino_rec_lb = fino_lb * (recup/100)
            valor_bloque = fino_rec_lb * precio
            
            st.markdown("### Resultados Financieros:")
            st.write(f"📦 Volumen: **{volumen:,.0f} m³**")
            st.write(f"⚖️ Tonelaje Mineral: **{tonelaje:,.0f} t**")
            st.write(f"🧱 Cobre Fino: **{fino_ton:.2f} t** ({fino_lb:,.0f} lbs)")
            st.divider()
            st.markdown(f"💵 **Valor In-Situ Estimado:** <span style='color:#00e676; font-size:2em'>US$ {valor_bloque:,.0f}</span>", unsafe_allow_html=True)

# ==============================================================================
# TAB 6: JORC
# ==============================================================================
with tabs[5]:
    if st.session_state['resultado']:
        res = st.session_state['resultado']
        
        st.markdown("### Clasificación de Recursos (JORC / NI 43-101)")
        
        if res['cat'] == "MEDIDO": css="jorc-medido"; icon="🟢"
        elif res['cat'] == "INDICADO": css="jorc-indicado"; icon="🟡"
        else: css="jorc-inferido"; icon="🔴"
        
        st.markdown(f"""
        <div class='jorc-card {css}'>
            <h1 style='margin:0; color:white;'>{icon} {res['cat']}</h1>
            <p>Coeficiente de Variación Kriging: <b>{res['cv_k']:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- NUEVA SECCIÓN EXPLICATIVA VISUAL ---
        st.markdown("### 📊 Gráfico de Certeza (Gauge Chart)")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = res['cv_k'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Coeficiente de Variación (CV)", 'font': {'size': 24, 'color': 'white'}},
            delta = {'reference': 15, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, max(50, res['cv_k'] + 10)], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "white"},
                'bgcolor': "#1a2332",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 15], 'color': "#1b5e20"},  # Medido
                    {'range': [15, 30], 'color': "#f57f17"}, # Indicado
                    {'range': [30, 1000], 'color': "#b71c1c"} # Inferido
                ],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="#0e1117", font={'color': "white"}, height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # --- EXPLICACIÓN DE RESULTADO ---
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Parámetros de Entrada:**")
            st.write(f"- Ley Estimada ($Z^*$): **{res['ley']:.4f}%**")
            st.write(f"- Desviación Estándar ($\sigma_k$): **{res['sigma']:.4f}**")
        with c2:
            st.markdown("**Cálculo del CV:**")
            st.latex(r"CV = \frac{\sigma_k}{Z^*} \times 100")
            st.markdown(f"$$CV = \\frac{{{res['sigma']:.4f}}}{{{res['ley']:.4f}}} \\times 100 = \\mathbf{{{res['cv_k']:.2f}\\%}}$$")
        
        st.markdown("---")
        st.markdown("**Veredicto del Auditor:**")
        if res['cv_k'] < 15:
            st.success(f"Como el CV ({res['cv_k']:.2f}%) es menor al 15%, el recurso se clasifica como **MEDIDO**.")
        elif res['cv_k'] <= 30:
            st.warning(f"Como el CV ({res['cv_k']:.2f}%) está entre 15% y 30%, el recurso se clasifica como **INDICADO**.")
        else:
            st.error(f"Como el CV ({res['cv_k']:.2f}%) es mayor al 30%, el recurso se clasifica como **INFERIDO**. Existe mucha incertidumbre.")

# ==============================================================================
# TAB 7: INFORME FINAL
# ==============================================================================
with tabs[6]:
    if st.session_state['resultado']:
        res = st.session_state['resultado']
        fecha = datetime.now().strftime("%d/%m/%Y")
        
        st.markdown("### 📄 Generador de Reporte Oficial")
        st.markdown("Haz clic abajo para descargar el informe completo en formato imprimible.")
        
        ic_90 = 1.645 * res['sigma']
        
        filas_html = ""
        for i in range(len(df_calc)):
            filas_html += f"""
            <tr>
                <td>{ids[i]}</td>
                <td>{res['d_vec'][i]:.2f}</td>
                <td>{res['M'][i]:.4f}</td>
                <td>{res['pesos'][i]:.4f}</td>
                <td>{df_calc['Ley'].values[i]:.2f}</td>
                <td><b>{(df_calc['Ley'].values[i] * res['pesos'][i]):.4f}</b></td>
            </tr>
            """

        reporte_html = f"""
        <div style="font-family: 'Arial', sans-serif; color: black; background: white; padding: 50px; border: 1px solid #ccc; max-width: 800px; margin: auto;">
            <div style="text-align:center; border-bottom: 2px solid #0277bd; padding-bottom: 10px; margin-bottom: 20px;">
                <h2 style="color: #0277bd; margin:0;">INFORME TÉCNICO DE RECURSOS</h2>
                <h4 style="margin:5px 0;">PROYECTO MINERO POMPERIA S.A.C.</h4>
            </div>
            
            <table style="width:100%; margin-bottom:20px;">
                <tr>
                    <td><b>Fecha:</b> {fecha}</td>
                    <td style="text-align:right;"><b>estudiante:</b> Felix Bautista Layme</td>
                </tr>
            </table>
            
            <h3 style="color:#333; border-bottom:1px solid #ccc;">1. RESUMEN EJECUTIVO</h3>
            <p>Se certifica la estimación de ley para el bloque de explotación ubicado en las coordenadas locales <b>X:{res['tx']}, Y:{res['ty']}</b>.</p>
            <ul style="background-color:#f5f5f5; padding:15px; border-left:5px solid #0277bd; list-style-type:none;">
                <li>💎 <b>Ley Estimada (Z*):</b> {res['ley']:.4f} %</li>
                <li>📉 <b>Varianza Kriging:</b> {res['var']:.4f}</li>
                <li>📋 <b>Categoría JORC:</b> {res['cat']}</li>
            </ul>
            
            <h3 style="color:#333; border-bottom:1px solid #ccc;">2. MEMORIA DE CÁLCULO</h3>
            <p>Detalle de la ponderación de leyes:</p>
            <table border="1" cellpadding="8" cellspacing="0" style="width:100%; border-collapse:collapse; text-align:center; font-size:0.9em;">
                <tr style="background:#0277bd; color:white;">
                    <th>Sondaje</th><th>Distancia (m)</th><th>Gamma (h)</th><th>Peso (λ)</th><th>Ley Real (%)</th><th>Aporte (%)</th>
                </tr>
                {filas_html}
                <tr style="background:#e0f7fa; font-weight:bold;">
                    <td colspan="5" style="text-align:right;">LEY ESTIMADA TOTAL (Z*):</td>
                    <td>{res['ley']:.4f}</td>
                </tr>
            </table>
            
            <h3 style="color:#333; border-bottom:1px solid #ccc;">3. ECONOMÍA Y RIESGO</h3>
            <p>Considerando un precio de <b>{precio} US$/lb</b>:</p>
            <ul>
                <li>Valor In-Situ del Bloque: <b>US$ {valor_bloque:,.2f}</b></li>
                <li>Incertidumbre (IC 90%): La ley real está entre <b>{res['ley'] - ic_90:.3f}%</b> y <b>{res['ley'] + ic_90:.3f}%</b>.</li>
            </ul>
            
            <h3 style="color:#333; border-bottom:1px solid #ccc;">4. CONCLUSIONES</h3>
            <p>El bloque ha sido clasificado como <b>{res['cat']}</b> con un coeficiente de variación del {res['cv_k']:.2f}%. 
            {"Se recomienda su incorporación inmediata al plan de minado." if res['cat'] != "INFERIDO" else "El riesgo es alto. Se requiere campaña de infill drilling."}</p>
            
            <br><br><br>
            <div style="text-align:center;">
                <p>__________________________</p>
                <p><b>Felix Bautista Layme</b><br>Firma del Responsable</p>
            </div>
        </div>
        """
        
        st.components.v1.html(reporte_html, height=800, scrolling=True)
        b64 = base64.b64encode(reporte_html.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="Informe_Tecnico_Oficial.html"><button style="background:#d32f2f; color:white; padding:12px 25px; border:none; border-radius:5px; cursor:pointer; font-size:1.1em; margin-top:10px;">📥 DESCARGAR INFORME OFICIAL (PDF)</button></a>'

        st.markdown(href, unsafe_allow_html=True)
