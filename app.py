import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.stats import norm, skew, kurtosis
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# 0. CONFIGURACIÓN INICIAL DEL SISTEMA
# ==============================================================================
# Configuración de la página debe ser la primera instrucción de Streamlit
st.set_page_config(
    page_title="GEOESTADISTICA MINERA - KRIGING PRO v2.0 (EDUCATIVO)", 
    page_icon="⚒️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. PROTOCOLO VISUAL (CSS AVANZADO & DARK MODE CORPORATIVO)
# ==============================================================================
# Se ha ampliado el CSS para incluir estilos específicos para la visualización
# de matrices, cajas de teoría pedagógica y alertas JORC.
st.markdown("""
    <style>
    /* --- IMPORTACIÓN DE FUENTES --- */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Segoe+UI:wght@400;600;800&display=swap');

    /* --- CONFIGURACIÓN GENERAL DEL BODY --- */
    .stApp {
        background-color: #0e1117; 
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* --- TIPOGRAFÍA Y ENCABEZADOS --- */
    h1 { color: #ffffff; font-weight: 800; font-size: 2.5rem; letter-spacing: -1px; }
    h2 { color: #90caf9; font-weight: 700; border-bottom: 2px solid #1e88e5; padding-bottom: 10px; }
    h3 { color: #e3f2fd; font-weight: 600; margin-top: 20px; }
    h4, h5, h6 { color: #b0bec5; font-family: 'Roboto Mono', monospace; }
    p, li, label, span { font-size: 16px; line-height: 1.6; }
    
    /* --- CAJAS EDUCATIVAS (TEORÍA - DOCENTE) --- */
    /* Estas cajas guían al estudiante paso a paso */
    .theory-box {
        background: linear-gradient(135deg, #1a2332 0%, #151922 100%); 
        border-left: 6px solid #00bcd4;
        padding: 25px; 
        border-radius: 12px; 
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4); 
        color: #e1f5fe;
        transition: transform 0.2s;
    }
    .theory-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 188, 212, 0.2);
    }
    .theory-title {
        color: #4dd0e1; 
        font-weight: 800; 
        font-size: 1.3em; 
        display: block; 
        margin-bottom: 12px;
        text-transform: uppercase; 
        letter-spacing: 1.5px;
        border-bottom: 1px solid rgba(77, 208, 225, 0.3);
        padding-bottom: 8px;
    }
    
    /* --- ALERTAS Y RESULTADOS (EXITO) --- */
    .result-box {
        background-color: #1b3a25;
        border-left: 6px solid #00e676;
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px;
        color: #e8f5e9;
        box-shadow: 0 4px 10px rgba(0,230,118,0.1);
    }

    /* --- CAJAS MATEMÁTICAS (EXPLICACIÓN DE FÓRMULAS - PASO A PASO) --- */
    .math-step {
        background-color: #263238;
        border: 1px solid #37474f;
        border-left: 5px solid #ffca28;
        padding: 20px; 
        border-radius: 8px; 
        margin-top: 15px;
        margin-bottom: 15px;
        color: #eceff1; 
        font-family: 'Roboto Mono', monospace;
        font-size: 0.95em;
    }
    .matrix-container {
        overflow-x: auto;
        padding: 10px;
        background-color: #121212;
        border-radius: 5px;
        margin-top: 10px;
    }

    /* --- SEMÁFORO JORC (CLASIFICACIÓN) --- */
    .jorc-card {
        padding: 30px; 
        border-radius: 15px; 
        text-align: center; 
        margin-bottom: 20px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .jorc-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 5px;
        background: rgba(255,255,255,0.3);
    }
    
    .jorc-medido {
        background: linear-gradient(145deg, #1b5e20, #2e7d32); 
        border: 2px solid #66bb6a; 
        color: #ffffff;
        box-shadow: 0 0 25px rgba(102, 187, 106, 0.5);
    }
    .jorc-indicado {
        background: linear-gradient(145deg, #e65100, #f57c00); 
        border: 2px solid #ffb74d; 
        color: #ffffff;
        box-shadow: 0 0 25px rgba(255, 183, 77, 0.5);
    }
    .jorc-inferido {
        background: linear-gradient(145deg, #b71c1c, #c62828); 
        border: 2px solid #ef5350; 
        color: #ffffff;
        box-shadow: 0 0 25px rgba(239, 83, 80, 0.5);
    }

    /* --- BOTONES PERSONALIZADOS --- */
    .stButton>button {
        background: linear-gradient(90deg, #0277bd 0%, #01579b 100%); 
        color: white; 
        border: none;
        border-radius: 8px; 
        height: 60px; 
        font-weight: 800; 
        font-size: 1.3em;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0288d1 0%, #0277bd 100%);
        box-shadow: 0 10px 20px rgba(2, 119, 189, 0.6);
        transform: translateY(-3px);
    }
    .stButton>button:active {
        transform: translateY(1px);
    }

    /* --- PESTAÑAS (TABS) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; 
        background-color: #0e1117;
        padding-bottom: 15px;
        border-bottom: 2px solid #374151;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f2937; 
        color: #b0bec5; 
        border: 1px solid #374151; 
        border-radius: 8px 8px 0 0;
        padding: 15px 30px;
        font-size: 1.1em;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #37474f;
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #263238; 
        border-top: 5px solid #00bcd4; 
        color: #ffffff;
        font-weight: bold;
    }
    
    /* --- TABLAS DATAFRAME --- */
    [data-testid="stDataFrame"] {
        border: 1px solid #374151;
        border-radius: 8px;
        background-color: #1a2332;
    }
    
    /* --- INPUTS --- */
    .stTextInput>div>div>input {
        background-color: #1f2937;
        color: white;
        border-radius: 5px;
    }
    .stNumberInput>div>div>input {
        background-color: #1f2937;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. GESTIÓN DE ESTADO, DATOS Y VARIABLES GLOBALES (BACKEND)
# ==============================================================================

# Inicialización de Datos por Defecto (Fall-back data)
if 'df_data' not in st.session_state:
    st.session_state['df_data'] = pd.DataFrame({
        'Id': ['DDH-101', 'DDH-102', 'DDH-103', 'DDH-104', 'DDH-105', 'DDH-106'], # <--- AQUÍ ESTABA EL ERROR (Decía 'ID')
        'X': [385250.0, 385275.0, 385300.0, 385320.0, 385260.0, 385310.0],
        'Y': [8245100.0, 8245125.0, 8245080.0, 8245150.0, 8245090.0, 8245140.0],
        'Ley': [0.85, 1.12, 0.72, 0.95, 0.65, 1.05]
    })

if 'resultado' not in st.session_state:
    st.session_state['resultado'] = None

# Variables de Sesión para Información del Proyecto y Estudiantes
if 'project_name' not in st.session_state:
    st.session_state['project_name'] = "PROYECTO ACADÉMICO MINA ESCUELA"
if 'student_names' not in st.session_state:
    st.session_state['student_names'] = ["Estudiante 1"]
if 'docente_name' not in st.session_state:
    st.session_state['docente_name'] = "Ing. Arturo R. Chayña Rodríguez" # DOCENTE FIJO OBLIGATORIO

# --- FUNCIONES DE BACKEND ---

def cargar_csv():
    """
    Carga, valida y normaliza el archivo CSV subido por el usuario.
    Incluye manejo de errores robusto para evitar caídas del sistema.
    """
    uploaded = st.session_state.uploader_key
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            # 1. Normalización: Eliminar espacios y capitalizar (ej: " ley " -> "Ley")
            cols = [c.strip().capitalize() for c in df.columns]
            df.columns = cols
            
            # 2. Validación de columnas críticas (X, Y, Ley)
            required_cols = {'X', 'Y', 'Ley'}
            if required_cols.issubset(df.columns):
                # Generar IDs si no existen para trazabilidad
                if 'Id' not in df.columns: 
                    df['Id'] = [f"MUESTRA-{i+1}" for i in range(len(df))]
                
                # Conversión de tipos segura
                df['Id'] = df['Id'].astype(str)
                df['X'] = pd.to_numeric(df['X'], errors='coerce')
                df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
                df['Ley'] = pd.to_numeric(df['Ley'], errors='coerce')
                
                # Eliminar filas con nulos generados por la conversión
                df = df.dropna(subset=['X', 'Y', 'Ley'])
                
                st.session_state['df_data'] = df
                st.toast("✅ Base de datos cargada y normalizada correctamente.", icon="💾")
            else:
                st.error(f"❌ Error de Formato: El archivo CSV debe contener obligatoriamente las columnas: {required_cols}")
                st.info("Por favor, revise que su CSV use punto (.) para decimales y coma (,) para separar columnas.")
        except Exception as e:
            st.error(f"Error crítico al leer el archivo: {str(e)}")

def guardar_historial(res):
    """
    Persistencia local de resultados para trazabilidad.
    Guarda cada cálculo exitoso en 'historial_proyecto.csv'.
    """
    archivo = 'historial_proyecto.csv'
    nuevo_registro = {
        'Fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Proyecto': st.session_state['project_name'],
        'X_Bloque': res['tx'], 
        'Y_Bloque': res['ty'],
        'Ley_Estimada': round(res['ley'], 4),
        'Varianza_Kriging': round(res['var'], 4),
        'CV_Kriging': round(res['cv_k'], 2),
        'Categoria_JORC': res['cat']
    }
    df_new = pd.DataFrame([nuevo_registro])
    
    # Lógica Append/Write
    try:
        if not os.path.exists(archivo):
            df_new.to_csv(archivo, index=False)
        else:
            df_new.to_csv(archivo, mode='a', header=False, index=False)
    except PermissionError:
        st.warning("⚠️ No se pudo guardar el historial. Cierre el archivo CSV si lo tiene abierto.")

# ==============================================================================
# 3. MOTOR MATEMÁTICO (GEOESTADÍSTICA PURA)
# ==============================================================================

def variograma_esferico(h, c0, c1, a):
    """
    Calcula el valor del variograma teórico usando el modelo Esférico.
    
    Args:
        h (array): Distancias.
        c0 (float): Efecto Pepita (Nugget).
        c1 (float): Meseta Parcial (Sill - Nugget).
        a (float): Rango (Alcance).
    
    Returns:
        array: Valores de gamma correspondientes.
    """
    h = np.atleast_1d(h)
    val = np.zeros_like(h)
    c_total = c0 + c1 
    
    # Caso 1: h > Rango (Meseta)
    mask_r = h > a
    val[mask_r] = c_total
    
    # Caso 2: h <= Rango (Curva esférica)
    mask_i = ~mask_r
    # Fórmula Esférica Clásica: C0 + C1 * [1.5(h/a) - 0.5(h/a)^3]
    val[mask_i] = c0 + c1 * (1.5 * (h[mask_i] / a) - 0.5 * (h[mask_i] / a)**3)
    
    # Caso 3: h = 0 (Por definición gamma(0)=0, aunque nugget sea > 0)
    val[h == 0] = 0
    return val

def resolver_kriging(df, target, c0, c1, a):
    """
    Resuelve el sistema de ecuaciones de Kriging Ordinario (OK).
    
    El sistema matricial es: [K] * [W] = [M]
    Donde:
        K: Matriz de varianzas entre muestras (más fila/columna Lagrange).
        W: Vector de pesos incógnita.
        M: Vector de varianzas muestra-bloque.
    """
    try:
        coords = df[['X', 'Y']].values
        leyes = df['Ley'].values
        n = len(coords)
        
        # 1. Matriz de Distancias (Euclidiana)
        # cdist calcula la distancia entre todos los pares de puntos
        dist_mat = cdist(coords, coords)
        dist_target = cdist(coords, [target]).flatten()
        
        # 2. Construcción de Matrices Kriging
        # Matriz K (n+1 x n+1) por el multiplicador de Lagrange
        K = np.zeros((n+1, n+1))
        M = np.zeros(n+1)
        
        # Llenado usando el modelo variográfico elegido
        K_vals = variograma_esferico(dist_mat, c0, c1, a)
        K[:n, :n] = K_vals
        # Condiciones de insesgo (suma de pesos = 1)
        K[n, :] = 1.0; K[:, n] = 1.0; K[n, n] = 0.0
        
        # Vector M (n+1)
        M_vals = variograma_esferico(dist_target, c0, c1, a)
        M[:n] = M_vals; M[n] = 1.0 
        
        # 3. Resolución del Sistema Lineal (Inversión Matricial)
        # Usamos solve que es numéricamente más estable que inv(K)
        W = np.linalg.solve(K, M)
        pesos = W[:n]
        mu = W[n] # Multiplicador de Lagrange
        
        # 4. Cálculo de Resultados Finales
        ley_est = np.sum(pesos * leyes)
        
        # Varianza de Kriging Ordinario: Sum(Wi * Gamma_i_Bloque) + mu
        var_krig = np.sum(pesos * M_vals) + mu
        
        # Control de errores numéricos (varianza negativa por precisión de float)
        if var_krig < 0: var_krig = 0
        
        sigma_k = np.sqrt(var_krig)
        
        # 5. Clasificación JORC / NI 43-101 (Criterio Simplificado por CV)
        # CV = (Desviación / Media) * 100
        cv_kriging = (sigma_k / ley_est * 100) if ley_est > 0 else 100
        
        # Slope of Regression (Calidad de estimación condicional)
        var_global = np.var(leyes, ddof=1) if len(leyes) > 1 else 1.0
        slope = 1.0 - (var_krig / var_global) if var_global > 0 else 0
        
        # Umbrales didácticos estándar
        if cv_kriging < 15: cat = "MEDIDO"
        elif 15 <= cv_kriging <= 30: cat = "INDICADO"
        else: cat = "INFERIDO"
        
        return {
            'status': 'OK', 
            'ley': ley_est, 'var': var_krig, 'sigma': sigma_k,
            'cv_k': cv_kriging, 'slope': slope, 'cat': cat,
            'pesos': pesos, 'mu': mu, 'K': K, 'M': M, 'W_raw': W,
            'd_mat': dist_mat, 'd_vec': dist_target
        }
    except np.linalg.LinAlgError:
        return {'status': 'ERROR', 'msg': "Error Matemático: Matriz Singular. Posiblemente hay muestras duplicadas en la misma ubicación (X, Y)."}
    except Exception as e:
        return {'status': 'ERROR', 'msg': str(e)}

# ==============================================================================
# 4. INTERFAZ GRÁFICA (FRONTEND)
# ==============================================================================

# --- BARRA LATERAL (SIDEBAR) MEJORADA ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.markdown("## 🏗️ PANEL DE CONTROL")
    st.markdown("---")
    
    # 1. Configuración del Proyecto (Editable)
    st.markdown("### 📝 Datos del Proyecto")
    st.session_state['project_name'] = st.text_input("Nombre del Proyecto:", value=st.session_state['project_name'])
    
    st.markdown("### 👨‍🎓 Equipo de Estudiantes")
    num_students = st.number_input("Número de integrantes", 1, 4, 1)
    
    student_list = []
    for i in range(num_students):
        student_list.append(st.text_input(f"Estudiante {i+1}:", value=st.session_state['student_names'][0] if i==0 else f"Estudiante {i+1}"))
    st.session_state['student_names'] = student_list
    
    st.markdown("---")
    st.markdown(f"**🎓 Docente:**\n{st.session_state['docente_name']}")
    st.markdown("**📅 Fecha:** " + datetime.now().strftime("%d/%m/%Y"))
    
    st.markdown("---")
    st.markdown("### 📚 GLOSARIO TÉCNICO")
    with st.expander("📖 Ver Definiciones (A-Z)"):
        st.markdown("""
        **A - Anisotropía:** Variabilidad distinta según la dirección.
        <hr style="margin:5px 0">
        **C - Covarianza:** Medida de correlación espacial.
        <hr style="margin:5px 0">
        **K - Kriging:** Estimador lineal insesgado óptimo (BLUE).
        <hr style="margin:5px 0">
        **N - Nugget (C0):** Variabilidad a muy corta distancia o error de muestreo.
        <hr style="margin:5px 0">
        **R - Rango (a):** Distancia donde las muestras dejan de tener correlación.
        <hr style="margin:5px 0">
        **S - Sill (Meseta):** Varianza total de la población.
        <hr style="margin:5px 0">
        **V - Varianza de Kriging:** Error de estimación asociado al bloque.
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("""
    **🚀 GUÍA DE USO:**
    Sigue las **7 Pestañas** en orden superior.
    
    ⚠️ **IMPORTANTE:** Al finalizar, ve a la pestaña **'7. Informe'** para descargar tu reporte final.
    """)
    st.success("✅ **Sistema en Línea**")
    st.markdown("<div style='text-align:center; color:#555; font-size:0.8em;'>v2.0 Build 2025</div>", unsafe_allow_html=True)

# --- CABECERA PRINCIPAL ---
st.title(f"{st.session_state['project_name']}")
st.markdown(f"#### Simulador de Estimación de Recursos Minerales con Kriging | Curso de Geoestadística Minera")

# Definición de Pestañas (Nombres cortos para que se vean todos en pantalla)
tabs = st.tabs([
    "📊 1. Datos", 
    "📈 2. Variograma", 
    "⚙️ 3. Kriging", 
    "🧮 4. Calculos",
    "💰 5. Economía",
    "⚖️ 6. JORC", 
    "📜 7. Informe"
])

# ==============================================================================
# TAB 1: ANÁLISIS DE DATOS (QA/QC)
# ==============================================================================
with tabs[0]:
    st.markdown("""
    <div class='theory-box'>
        <span class='theory-title'>🔍 Módulo 1: Validación y Análisis Exploratorio de Datos (EDA)</span>
        <p>Antes de realizar cualquier estimación, el <b>Ingeniero Geoestadístico</b> debe auditar sus datos ("Conoce tus datos"). 
        Buscamos valores atípicos (outliers), errores de coordenadas y entendemos la distribución estadística.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.subheader("📥 Carga de Base de Datos")
        
        with st.expander("ℹ️ Instrucciones y Formato CSV"):
            st.markdown("""
            Para usar tus propios datos, sube un archivo **.csv** con las siguientes columnas (el orden no importa, pero los nombres sí):
            | X | Y | Ley | ID (Opcional) |
            |---|---|---|---|
            | 100 | 200 | 1.5 | M-1 |
            | 110 | 210 | 2.1 | M-2 |
            
            *Nota: Usa punto (.) para decimales.*
            """)
        
        st.file_uploader("Arrastra tu archivo aquí:", type=['csv'], key="uploader_key", on_change=cargar_csv)
        
        st.markdown("### 📋 Vista Previa de Datos")
        st.dataframe(st.session_state['df_data'], height=300, use_container_width=True)
        st.info(f"Total de Muestras: **{len(st.session_state['df_data'])}**")

    with c2:
        st.subheader("📊 Estadística Descriptiva y Gráficos")
        df = st.session_state['df_data']
        
        # Limpieza interna para cálculos
        df_calc = df.copy()
        cols_numericas = ['X', 'Y', 'Ley']
        for col in cols_numericas:
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')
        df_calc = df_calc.dropna(subset=cols_numericas)
        
        if not df_calc.empty:
            # Cálculos Estadísticos Avanzados
            media = df_calc['Ley'].mean()
            mediana = df_calc['Ley'].median()
            std = df_calc['Ley'].std()
            min_val = df_calc['Ley'].min()
            max_val = df_calc['Ley'].max()
            var = df_calc['Ley'].var()
            kurt = kurtosis(df_calc['Ley'])
            skewness = skew(df_calc['Ley'])
            cv = (std/media*100) if media>0 else 0
            
            # --- TARJETAS MÉTRICAS ---
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Media (Ley)", f"{media:.2f} %", help="Promedio aritmético de las leyes")
            col_m2.metric("Desviación Std.", f"{std:.2f}", help="Dispersión de los datos respecto a la media")
            col_m3.metric("Coef. Variación", f"{cv:.1f} %", delta="Alto Riesgo" if cv>100 else "Estable", delta_color="inverse")
            col_m4.metric("Máximo", f"{max_val:.2f} %")
            
            st.markdown("---")
            
            # --- INTERPRETACIÓN DOCENTE ---
            st.markdown(f"""
            <div class='math-step'>
                <b>🧠 Interpretación Docente:</b><br>
                <ul>
                    <li>El <b>Coeficiente de Variación (CV)</b> es {cv:.2f}%. 
                        {"Si es < 50%, la distribución es regular y fácil de estimar." if cv < 50 else 
                         "Si está entre 50-100%, requiere cuidado. Si es > 100%, indica presencia de 'Pepitas' (valores extremos) que pueden sesgar el Kriging."}
                    </li>
                    <li><b>Sesgo (Skewness):</b> {skewness:.2f}. {"Valor positivo indica cola a la derecha (muchas leyes bajas, pocas altas)." if skewness > 0 else "Valor negativo indica cola a la izquierda."}</li>
                    <li><b>Curtosis:</b> {kurt:.2f}. Indica qué tan 'puntiaguda' es la distribución.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # --- PESTAÑAS GRÁFICAS ---
            t1, t2, t3 = st.tabs(["📊 Histograma & Boxplot", "🗺️ Mapa de Ubicación", "📈 Derivas (Tendencias)"])
            
            with t1:
                # Histograma y Boxplot combinados
                fig_dist = px.histogram(
                    df_calc, x="Ley", nbins=15, marginal="box", 
                    title="Distribución de Frecuencias de Ley",
                    color_discrete_sequence=['#00bcd4'],
                    hover_data=df_calc.columns
                )
                fig_dist.add_vline(x=media, line_dash="dash", line_color="red", annotation_text="Media")
                fig_dist.update_layout(template="plotly_dark", height=350, bargap=0.1)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with t2:
                # Mapa de Ubicación (Scatter Plot)
                fig_map = px.scatter(
                    df_calc, x='X', y='Y', size='Ley', color='Ley',
                    hover_name='Id', title="Mapa de Ubicación de Sondajes (Planta)",
                    color_continuous_scale='Viridis', size_max=40
                )
                fig_map.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig_map, use_container_width=True)

            with t3:
                # Análisis de Deriva (Drift Analysis)
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    fig_dx = px.scatter(df_calc, x='X', y='Ley', trendline="ols", title="Deriva Este-Oeste", trendline_color_override="red")
                    fig_dx.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig_dx, use_container_width=True)
                with c_d2:
                    fig_dy = px.scatter(df_calc, x='Y', y='Ley', trendline="ols", title="Deriva Norte-Sur", trendline_color_override="red")
                    fig_dy.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig_dy, use_container_width=True)
        else:
            st.warning("⚠️ No hay datos válidos para procesar.")

# ==============================================================================
# TAB 2: VARIOGRAFÍA ESTRUCTURAL
# ==============================================================================
with tabs[1]:
    st.markdown("""
    <div class='theory-box'>
        <span class='theory-title'>📈 Módulo 2: Modelamiento del Variograma</span>
        <p>El variograma es la herramienta fundamental de la Geoestadística. Nos dice <b>qué tan parecidas son dos muestras</b> en función de la distancia que las separa.
        Debemos ajustar la curva teórica (Azul) a la realidad geológica del yacimiento.</p>
    </div>
    """, unsafe_allow_html=True)
    
    cv1, cv2 = st.columns([1, 2.5])
    
    with cv1:
        st.subheader("🛠️ Ajuste de Parámetros")
        st.markdown("Modifique estos valores para ajustar el modelo:")
        
        v_c0 = st.number_input("1️⃣ Efecto Pepita (C0 - Nugget)", 0.0, 50.0, 0.015, step=0.001, format="%.3f", help="Error aleatorio a distancia cero.")
        v_c1 = st.number_input("2️⃣ Meseta Parcial (C1)", 0.0, 100.0, 0.085, step=0.001, format="%.3f", help="Varianza estructurada.")
        v_a  = st.number_input("3️⃣ Rango / Alcance (a)", 1.0, 2000.0, 120.0, step=10.0, format="%.1f", help="Distancia máxima de correlación.")
        
        meseta_total = v_c0 + v_c1
        st.info(f"🔢 **Meseta Total (Sill):** {meseta_total:.3f}")
        
        st.markdown("---")
        st.markdown("""
        **Guía Rápida:**
        * **Alto C0:** Muestreo errático.
        * **Rango Corto:** Mineralización discontinua.
        * **Rango Largo:** Mineralización continua y homogénea.
        """)
    
    with cv2:
        # Generación de datos para el gráfico
        h = np.linspace(0, v_a * 1.5, 100)
        gamma = variograma_esferico(h, v_c0, v_c1, v_a)
        
        fig_var = go.Figure()
        
        # Curva del Modelo
        fig_var.add_trace(go.Scatter(x=h, y=gamma, mode='lines', name='Modelo Esférico', line=dict(color='#00bcd4', width=5)))
        
        # Líneas de Referencia (Anotaciones Didácticas)
        fig_var.add_hline(y=meseta_total, line_dash="dash", line_color="green", annotation_text="Meseta (Sill)", annotation_position="top right")
        fig_var.add_vline(x=v_a, line_dash="dash", line_color="orange", annotation_text="Rango (a)", annotation_position="bottom right")
        
        # Anotación Nugget
        fig_var.add_annotation(x=0, y=v_c0, text="Nugget (C0)", showarrow=True, arrowhead=2, ax=40, ay=-40, font=dict(color="yellow"))
        
        fig_var.update_layout(
            title="Variograma Teórico Ajustado",
            xaxis_title="Distancia de Separación (h) [metros]",
            yaxis_title="Variabilidad - Gamma (γ)",
            template="plotly_dark",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_var, use_container_width=True)

# ==============================================================================
# TAB 3: ESTIMACIÓN (KRIGING)
# ==============================================================================
with tabs[2]:
    st.markdown("""
    <div class='theory-box'>
        <span class='theory-title'>⚙️ Módulo 3: Estimación de Bloque (Interpolación)</span>
        <p>Defina las coordenadas del centro del bloque a estimar. El algoritmo buscará las muestras cercanas, 
        asignará pesos óptimos basados en el variograma (Tab 2) y calculará la ley más probable.</p>
    </div>
    """, unsafe_allow_html=True)

    c_izq, c_der = st.columns([1, 2])
    
    with c_izq:
        st.subheader("📍 Coordenadas del Bloque")
        # Pre-cargar valores centrales de los datos
        default_x = df_calc['X'].mean()
        default_y = df_calc['Y'].mean()
        
        tx = st.number_input("Coordenada Este (X)", value=float(round(default_x, 0)))
        ty = st.number_input("Coordenada Norte (Y)", value=float(round(default_y, 0)))
        
        st.divider()
        
        col_btn, col_info = st.columns([2, 1])
        if st.button("🚀 EJECUTAR KRIGING"):
            with st.spinner('Resolviendo sistema matricial...'):
                res = resolver_kriging(df_calc, [tx, ty], v_c0, v_c1, v_a)
                if res['status'] == 'OK':
                    # Añadimos metadatos al resultado
                    res.update({'tx': tx, 'ty': ty, 'c0': v_c0, 'c1': v_c1, 'a': v_a, 'fecha': datetime.now()})
                    st.session_state['resultado'] = res
                    guardar_historial(res)
                    st.success("¡Cálculo Exitoso!")
                else:
                    st.error(res['msg'])
                    st.session_state['resultado'] = None

    with c_der:
        if st.session_state['resultado'] and st.session_state['resultado']['status']=='OK':
            res = st.session_state['resultado']
            
            # --- PANEL DE RESULTADOS DESTACADO ---
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937 0%, #111827 100%); border:2px solid #00e676; border-radius:15px; padding:25px; text-align:center; box-shadow: 0 0 20px rgba(0, 230, 118, 0.2);">
                <h4 style="color:#00e676; margin:0; letter-spacing: 2px;">LEY ESTIMADA (Z*)</h4>
                <h1 style="color:white; font-size:4.5em; margin:10px 0; text-shadow: 0 0 10px rgba(255,255,255,0.3);">{res['ley']:.4f} %</h1>
                <div style="display:flex; justify-content:space-around; margin-top:20px; border-top: 1px solid #374151; padding-top: 15px;">
                    <div>
                        <span style="color:#b0bec5; font-size:0.9em;">Varianza de Estimación ($\sigma_k^2$)</span><br>
                        <b style="color:#b3e5fc; font-size:1.4em;">{res['var']:.4f}</b>
                    </div>
                    <div>
                        <span style="color:#b0bec5; font-size:0.9em;">Desviación Estándar ($\sigma_k$)</span><br>
                        <b style="color:#b3e5fc; font-size:1.4em;">{res['sigma']:.4f}</b>
                    </div>
                    <div>
                        <span style="color:#b0bec5; font-size:0.9em;">Pendiente (Slope)</span><br>
                        <b style="color:#ffcc80; font-size:1.4em;">{res['slope']:.4f}</b>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- VISUALIZACIÓN 2D DEL BLOQUE Y MUESTRAS ---
            # Detectamos si la columna se llama 'Id' o 'ID' automáticamente
            col_id = 'Id' if 'Id' in df_calc.columns else 'ID'

            fig_plan = px.scatter(df_calc, x='X', y='Y', size='Ley', color='Ley', 
                                title=f"Plano de Estimación (Bloque en X:{tx:.1f}, Y:{ty:.1f})", 
                                color_continuous_scale='Viridis',
                                hover_data=[col_id]) # <--- Aquí usamos la columna detectada
            
            # Añadir el bloque como un marcador distinto
            fig_plan.add_trace(go.Scatter(
                x=[tx], y=[ty], mode='markers+text', 
                marker=dict(color='#ff1744', size=30, symbol='square', line=dict(color='white', width=2)), 
                name='BLOQUE A ESTIMAR', text=["BLOQUE"], textposition="top center",
                textfont=dict(size=14, color="white", family="Arial Black")
            ))
            
            # Añadir Radio de Influencia
            t = np.linspace(0, 2*np.pi, 100)
            fig_plan.add_trace(go.Scatter(
                x=tx+v_a*np.cos(t), y=ty+v_a*np.sin(t), 
                mode='lines', line=dict(dash='dash', color='white', width=1), 
                name='Radio de Influencia (Rango)'
            ))
            
            fig_plan.update_layout(template="plotly_dark", height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_plan, use_container_width=True)

# ==============================================================================
# TAB 4: INGENIERÍA INVERSA (BLACK BOX REVEALED) - SECCIÓN CLAVE
# ==============================================================================
with tabs[3]:
    if st.session_state['resultado']:
        res = st.session_state['resultado']
        st.markdown("""
        <div class='theory-box'>
            <span class='theory-title'>🧮 Módulo 4: "Caja Blanca" - Desglose Matemático</span>
            <p>Aquí abrimos el algoritmo para fines docentes. Observará cómo se calculan las distancias, 
            se construye el sistema matricial <b>[K] * [W] = [M]</b> y se obtiene el peso de cada muestra.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- PASO 1: GEOMETRÍA ---
        st.markdown("### 🔹 Paso 1: Cálculo de Distancias Geométricas")
        st.write("Calculamos la distancia euclidiana ($d$) desde cada sondaje hasta el centro del bloque.")
        
        ids = df_calc['Id'].tolist() if 'Id' in df_calc.columns else [str(i) for i in range(len(df_calc))]
        
        # Tabla detallada Paso 1
        data_step1 = []
        for i in range(len(df_calc)):
            dist = res['d_vec'][i]
            dx = df_calc['X'].iloc[i] - res['tx']
            dy = df_calc['Y'].iloc[i] - res['ty']
            data_step1.append({
                "ID": ids[i],
                "Este (X)": df_calc['X'].iloc[i],
                "Norte (Y)": df_calc['Y'].iloc[i],
                "ΔX": dx, "ΔY": dy,
                "Distancia (m)": dist
            })
        st.dataframe(pd.DataFrame(data_step1).style.format({"Este (X)": "{:.2f}", "Norte (Y)": "{:.2f}", "ΔX": "{:.1f}", "ΔY": "{:.1f}", "Distancia (m)": "{:.3f}"}))

        # --- PASO 2: VARIOGRAFÍA APLICADA ---
        st.markdown("### 🔹 Paso 2: Conversión a Varianzas (Gamma)")
        st.write(f"Usando el modelo ajustado (C0={res['c0']}, C1={res['c1']}, a={res['a']}), transformamos las distancias en valores de Gamma $\gamma(h)$.")
        
        data_step2 = []
        for i in range(len(df_calc)):
            data_step2.append({
                "ID": ids[i],
                "Distancia al Bloque (h)": res['d_vec'][i],
                "Gamma Bloque γ(h)": res['M'][i] # Estos son los valores del vector M (lado derecho)
            })
        # Solo aplicamos formato de decimales a las columnas numéricas, NO al ID
        st.dataframe(pd.DataFrame(data_step2).style.format(
            subset=["Distancia al Bloque (h)", "Gamma Bloque γ(h)"], 
            formatter="{:.4f}"
        ))

        # --- PASO 3: SISTEMA MATRICIAL (VISUALIZACIÓN AVANZADA) ---
        st.markdown("### 🔹 Paso 3: Resolución del Sistema de Kriging")
        st.markdown("El sistema de ecuaciones lineales es:")
        st.latex(r"[K] \cdot [W] = [M]")
        
        col_mat1, col_mat2 = st.columns(2)
        with col_mat1:
            st.info("Donde [K] es la matriz de covarianzas entre muestras (+ Lagrange):")
            # Visualizar matriz K si no es gigante
            if len(df_calc) <= 10:
                st.write(pd.DataFrame(res['K'], columns=ids+['μ'], index=ids+['μ']).style.background_gradient(cmap='Blues', axis=None).format("{:.3f}"))
            else:
                st.warning("Matriz K es muy grande para visualizar completa (N > 10).")
        
        with col_mat2:
            st.info("Donde [M] es la covarianza Muestra-Bloque:")
            st.write(pd.DataFrame(res['M'], index=ids+['μ'], columns=['Vector M']).style.background_gradient(cmap='Greens').format("{:.3f}"))

        # --- PASO 4: PONDERACIÓN Y RESULTADO ---
        st.markdown("### 🔹 Paso 4: Obtención de Pesos y Ley Final")
        
        st.markdown("""
        <div class='math-step'>
            Al resolver el sistema matricial, obtenemos los pesos ($\lambda$). 
            Luego, la Ley Final ($Z^*$) es la suma ponderada de cada ley por su peso:
        </div>
        """, unsafe_allow_html=True)

        # Fórmulas matemáticas renderizadas correctamente
        c_mat1, c_mat2 = st.columns(2)
        with c_mat1:
            st.markdown("**1. Solución Matricial:**")
            st.latex(r"[W] = [K]^{-1} \cdot [M]")
        with c_mat2:
            st.markdown("**2. Ecuación de Estimación:**")
            st.latex(r"Z^* = \sum_{i=1}^{n} \lambda_i \cdot Z(x_i)")

        leyes_reales = df_calc['Ley'].values
        pesos_calc = res['pesos']
        aportes = leyes_reales * pesos_calc
        
        df_final_weights = pd.DataFrame({
            'Sondaje': ids,
            'Ley Real (Z)': leyes_reales,
            'Peso Kriging (λ)': pesos_calc,
            'Aporte (λ * Z)': aportes
        })
        
        # Resaltar pesos negativos (Screening effect)
        def highlight_neg(val):
            color = 'red' if val < 0 else 'lightgreen'
            return f'color: {color}; font-weight: bold'
            
        # CORRECCIÓN: Aplicamos el formato de 4 decimales SOLO a las columnas numéricas
        st.dataframe(df_final_weights.style.applymap(highlight_neg, subset=['Peso Kriging (λ)']).format(
            subset=['Ley Real (Z)', 'Peso Kriging (λ)', 'Aporte (λ * Z)'], 
            formatter="{:.4f}"
        ))
        
        # Suma final explicita
        suma_aportes = np.sum(aportes)
        st.markdown(f"#### ✅ Suma de Aportes = **{suma_aportes:.4f} %** (Coincide con la Ley Estimada)")
    else:
        st.info("⚠️ Primero ejecute la estimación en la pestaña 3.")

# ==============================================================================
# TAB 5: ECONOMÍA MINERA
# ==============================================================================
with tabs[4]:
    if st.session_state['resultado']:
        res = st.session_state['resultado']
        st.markdown("""
        <div class='theory-box'>
            <span class='theory-title'>💰 Módulo 5: Valorización Económica del Bloque</span>
            <p>Un ingeniero de minas no solo estima leyes, estima <b>dinero</b>. Aquí transformamos la variable geológica (%) 
            en valor monetario (US$), considerando tonelaje, precios y recuperaciones.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Inputs Económicos
        ce1, ce2, ce3 = st.columns(3)
        with ce1:
            st.markdown("**1. Geometría del Bloque**")
            dim_x = st.number_input("Largo (m)", value=20.0)
            dim_y = st.number_input("Ancho (m)", value=20.0)
            dim_z = st.number_input("Alto (Banco) (m)", value=10.0)
        with ce2:
            st.markdown("**2. Parámetros Físicos**")
            densidad = st.number_input("Densidad (t/m³)", value=2.60)
            recup = st.number_input("Recuperación Metalúrgica (%)", value=88.0)
        with ce3:
            st.markdown("**3. Mercado**")
            precio = st.number_input("Precio del Metal (US$/lb)", value=4.15)
            costo_minado = st.number_input("Costo Op. Total (US$/t)", value=45.0)

        # Cálculos Económicos
        volumen = dim_x * dim_y * dim_z
        tonelaje = volumen * densidad
        fino_ton = tonelaje * (res['ley']/100)
        fino_lbs = fino_ton * 2204.62
        fino_recuperado_lbs = fino_lbs * (recup/100)
        
        ingreso_bruto = fino_recuperado_lbs * precio
        costo_total_bloque = tonelaje * costo_minado
        profit = ingreso_bruto - costo_total_bloque
        
        st.divider()
        
        # --- VISUALIZACIÓN DE RESULTADOS FINANCIEROS ---
        kf1, kf2 = st.columns([1, 1.5])
        
        with kf1:
            st.markdown("### 🧾 Balance Financiero")
            st.write(f"📦 **Volumen:** {volumen:,.0f} m³")
            st.write(f"⚖️ **Tonelaje:** {tonelaje:,.0f} t")
            st.write(f"🧱 **Cobre Fino:** {fino_ton:.2f} t ({fino_lbs:,.0f} lbs)")
            st.markdown("---")
            st.write(f"💵 **Ingresos (NSR):** US$ {ingreso_bruto:,.2f}")
            st.write(f"📉 **Costos:** US$ {costo_total_bloque:,.2f}")
            
            # Resultado Final Grande
            color_res = "#00e676" if profit > 0 else "#ff1744"
            st.markdown(f"### Beneficio Neto:")
            st.markdown(f"<span style='color:{color_res}; font-size:2.5em; font-weight:bold;'>US$ {profit:,.2f}</span>", unsafe_allow_html=True)

        with kf2:
            # Gráfico de Cascada (Waterfall) o Pie Chart
            fig_fin = go.Figure(data=[go.Pie(
                labels=['Costo Operativo', 'Beneficio Neto' if profit > 0 else 'Pérdida'], 
                values=[costo_total_bloque, abs(profit)],
                hole=.4,
                marker_colors=['#ef5350', '#66bb6a' if profit > 0 else '#b71c1c']
            )])
            fig_fin.update_layout(title="Distribución del Valor Económico del Bloque", template="plotly_dark")
            st.plotly_chart(fig_fin, use_container_width=True)
            
            # Análisis de Sensibilidad Rápido
            st.info(f"El bloque paga sus costos si el precio es > US$ {(costo_total_bloque / fino_recuperado_lbs):.2f} /lb")
    else:
        st.info("⚠️ Ejecute primero la estimación.")

# ==============================================================================
# TAB 6: CLASIFICACIÓN JORC / NI 43-101
# ==============================================================================
with tabs[5]:
    if st.session_state['resultado']:
        res = st.session_state['resultado']
        
        st.markdown("""
        <div class='theory-box'>
            <span class='theory-title'>⚖️ Módulo 6: Clasificación de Recursos (Estándar Internacional)</span>
            <p>Para reportar recursos a la bolsa de valores (JORC en Australia, NI 43-101 en Canadá), debemos clasificar la confianza.
            Usamos el <b>Error Relativo (Coeficiente de Variación del Kriging)</b> como proxy de la incertidumbre.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Determinar estilos según categoría
        if res['cat'] == "MEDIDO": 
            css_class="jorc-medido"
            icon="🟢"
            msg_auditor = "Alta confianza geológica. Se permite planificación minera detallada y conversión a Reservas Probadas."
        elif res['cat'] == "INDICADO": 
            css_class="jorc-indicado"
            icon="🟡"
            msg_auditor = "Confianza razonable. Permite planificación general y conversión a Reservas Probables."
        else: 
            css_class="jorc-inferido"
            icon="🔴"
            msg_auditor = "Baja confianza. Solo para evaluación preliminar. NO se puede convertir a Reservas ni usar en plan minero."
        
        # Tarjeta Principal JORC
        st.markdown(f"""
        <div class='jorc-card {css_class}'>
            <h2 style='color:white; margin:0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>{icon} RECURSO {res['cat']}</h2>
            <h4 style="color:white; margin-top:10px;">Coeficiente de Variación (CV): {res['cv_k']:.2f}%</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_j1, col_j2 = st.columns([1, 1])
        
        with col_j1:
            st.markdown("### 📉 Gráfico de Incertidumbre")
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = res['cv_k'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Error Relativo de Estimación (%)", 'font': {'size': 20}},
                delta = {'reference': 15, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "white", 'thickness': 0.3},
                    'bgcolor': "#121212",
                    'steps': [
                        {'range': [0, 15], 'color': "#2e7d32"},   # Medido
                        {'range': [15, 30], 'color': "#ef6c00"},  # Indicado
                        {'range': [30, 100], 'color': "#c62828"}   # Inferido
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': res['cv_k']
                    }
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20,r=20,t=50,b=20), paper_bgcolor="#0e1117", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_j2:
            st.markdown("### 📋 Criterios del Código")
            st.table(pd.DataFrame({
                'Categoría': ['MEDIDO', 'INDICADO', 'INFERIDO'],
                'CV Kriging (%)': ['< 15%', '15% - 30%', '> 30%'],
                'Nivel de Riesgo': ['Bajo', 'Moderado', 'Alto']
            }).set_index('Categoría'))

            # --- NUEVA EXPLICACIÓN DIDÁCTICA DEL CV ---
            st.markdown("---")
            st.subheader("🧮 Detalle del Cálculo: Coeficiente de Variación (CV)")
            st.markdown("El CV mide la incertidumbre relativa. Se calcula dividiendo la desviación estándar del Kriging entre la ley estimada.")
            
            # Fórmula general
            st.latex(r"CV (\%) = \left( \frac{\sigma_{kriging}}{Z^*_{estimado}} \right) \times 100")
            
            # Reemplazo con números reales
            st.info(f"""
            **Reemplazando con tus datos:**
            
            $$ CV = \\frac{{{res['sigma']:.4f}}}{{{res['ley']:.4f}}} \\times 100 = \\mathbf{{{res['cv_k']:.2f}\\%}} $$
            
            *Interpretación: El error es el {res['cv_k']:.2f}% del valor estimado.*
            """)
            
            st.markdown(f"""
            <div class='result-box'>
                <b>👨‍⚖️ Veredicto del Auditor:</b><br>
                {msg_auditor}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("⚠️ Ejecute primero la estimación.")

# ==============================================================================
# TAB 7: INFORME (CORREGIDO Y BLINDADO)
# ==============================================================================
with tabs[6]:
    if st.session_state['resultado']:
        res = st.session_state['resultado']
        
        st.markdown("### 📄 Generador de Reporte Técnico")
        
        # 1. Lista de estudiantes
        est_li = "".join([f"<li>{e}</li>" for e in estudiantes_activos])
        
        # 2. LÓGICA DE SEGURIDAD (ESTA ES LA CURA AL ERROR)
        # Calcula el número menor de filas para que nunca se desborde, tengas 6 o 47 datos.
        df_safe = st.session_state['df_data']
        
        # Aseguramos que las listas tengan el mismo largo antes de iterar
        limit = min(len(res['pesos']), len(res['d_vec']), len(df_safe))
        
        ids = df_safe['ID'].values
        leys = df_safe['Ley'].values
        
        # 3. Generación de filas seguras
        rows = ""
        for i in range(limit):
            try:
                # Extraemos valores con seguridad dentro del límite
                id_val = str(ids[i])
                dist_val = float(res['d_vec'][i])
                peso_val = float(res['pesos'][i])
                ley_val = float(leys[i])
                
                rows += f"""
                <tr>
                    <td>{id_val}</td>
                    <td>{dist_val:.2f}</td>
                    <td>{peso_val:.4f}</td>
                    <td>{ley_val:.2f}</td>
                </tr>"""
            except:
                continue

        # 4. El HTML del Reporte
        html = f"""
        <div style="font-family:Arial; padding:40px; background:white; color:black; border:1px solid #ccc;">
            <center>
                <h1 style="color:#0277bd;">INFORME DE ESTIMACIÓN DE RECURSOS</h1>
                <h3>PROYECTO: {proj_name.upper()}</h3>
            </center>
            <hr>
            <table width="100%">
                <tr>
                    <td><b>Docente:</b> Ing. Arturo R. Chayña Rodriguez</td>
                    <td align="right"><b>Fecha:</b> {datetime.now().strftime('%d/%m/%Y')}</td>
                </tr>
                <tr><td colspan="2"><b>Equipo Técnico:</b><ul>{est_li}</ul></td></tr>
            </table>
            
            <h3>1. RESUMEN EJECUTIVO</h3>
            <div style="background:#e3f2fd; padding:15px; border-radius:5px;">
                <p>Estimación del Bloque en <b>X={res['tx']:.2f}, Y={res['ty']:.2f}</b>:</p>
                <ul>
                    <li><b>LEY ESTIMADA: {res['ley']:.4f} % Cu</b></li>
                    <li>Varianza Kriging: {res['var']:.4f}</li>
                    <li>Categoría: <b>{res['cat']}</b></li>
                </ul>
            </div>
            
            <h3>2. DETALLE DE MUESTRAS ({limit} registros procesados)</h3>
            <table border="1" cellspacing="0" cellpadding="5" width="100%">
                <tr style="background:#0277bd; color:white;"><th>ID</th><th>Distancia (m)</th><th>Peso</th><th>Ley</th></tr>
                {rows}
            </table>
            
            <br><br><br>
            <center>
                <p>________________________________________________</p>
                <p><b>RESPONSABLE FACULTAD DE INGENIERÍA DE MINAS - UNA PUNO</b></p>
                <p>Semestre 2025 - II</p>
            </center>
        </div>
        """
        
        st.components.v1.html(html, height=700, scrolling=True)
        b64 = base64.b64encode(html.encode()).decode()
        st.markdown(f'<a href="data:text/html;base64,{b64}" download="Reporte_{proj_name}.html"><button style="background-color:#2e7d32; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer;">📥 DESCARGAR INFORME OFICIAL</button></a>', unsafe_allow_html=True)
