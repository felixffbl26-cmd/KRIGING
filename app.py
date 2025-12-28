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
from informe_generator import generar_informe_html

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
    Soporta archivos grandes (100+ muestras).
    """
    uploaded = st.session_state.uploader_key
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            
            # Información sobre el tamaño del archivo
            num_rows = len(df)
            if num_rows > 100:
                st.info(f"📊 Archivo grande detectado: {num_rows} muestras. El procesamiento puede tomar unos segundos...")
            
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
                rows_before = len(df)
                df = df.dropna(subset=['X', 'Y', 'Ley'])
                rows_after = len(df)
                
                if rows_before > rows_after:
                    st.warning(f"⚠️ Se eliminaron {rows_before - rows_after} filas con datos inválidos.")
                
                st.session_state['df_data'] = df
                st.toast(f"✅ Base de datos cargada: {rows_after} muestras válidas procesadas correctamente.", icon="💾")
                
                # Mensaje adicional para datasets grandes
                if rows_after > 200:
                    st.success(f"🚀 ¡Excelente! El sistema puede manejar {rows_after} muestras sin problemas. La estimación Kriging usará todas las muestras disponibles.")
            else:
                st.error(f"❌ Error de Formato: El archivo CSV debe contener obligatoriamente las columnas: {required_cols}")
                st.info("Por favor, revise que su CSV use punto (.) para decimales y coma (,) para separar columnas.")
        except Exception as e:
            st.error(f"Error crítico al leer el archivo: {str(e)}")
            st.info("💡 Sugerencia: Verifique que el archivo sea un CSV válido y no esté corrupto.")


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

def variograma_gaussiano(h, c0, c1, a):
    """
    Modelo Gaussiano - Más suave, sin punto de inflexión.
    Útil para variables muy continuas.
    """
    h = np.atleast_1d(h)
    c_total = c0 + c1
    val = np.where(h == 0, 0, c0 + c1 * (1 - np.exp(-3 * (h/a)**2)))
    return val

def variograma_exponencial(h, c0, c1, a):
    """
    Modelo Exponencial - Alcanza la meseta asintóticamente.
    Común en depósitos sedimentarios.
    """
    h = np.atleast_1d(h)
    c_total = c0 + c1
    val = np.where(h == 0, 0, c0 + c1 * (1 - np.exp(-3 * h/a)))
    return val

def variograma_potencial(h, c0, c1, a):
    """
    Modelo de Potencia - Sin meseta definida.
    Para variables con tendencia o deriva.
    """
    h = np.atleast_1d(h)
    # Evitar división por cero
    val = np.where(h == 0, 0, c0 + c1 * (h/a)**a)
    return val

def calcular_variograma_experimental(df, n_lags=15, lag_tolerance=0.5):
    """
    Calcula el variograma experimental de los datos reales.
    
    Args:
        df: DataFrame con columnas X, Y, Ley
        n_lags: Número de intervalos de distancia
        lag_tolerance: Tolerancia del intervalo (fracción)
    
    Returns:
        dict con 'lags' (distancias) y 'gamma' (semivarianza)
    """
    coords = df[['X', 'Y']].values
    leyes = df['Ley'].values
    n = len(coords)
    
    # Calcular todas las distancias y diferencias al cuadrado
    dist_mat = cdist(coords, coords)
    diff_sq = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            diff_sq[i, j] = (leyes[i] - leyes[j])**2
            diff_sq[j, i] = diff_sq[i, j]
    
    # Determinar rango de distancias
    max_dist = np.max(dist_mat) / 2  # Usar solo hasta la mitad de la distancia máxima
    lag_size = max_dist / n_lags
    
    lags = []
    gammas = []
    counts = []
    
    for i in range(1, n_lags + 1):
        lag_center = i * lag_size
        lag_min = lag_center - lag_size * lag_tolerance
        lag_max = lag_center + lag_size * lag_tolerance
        
        # Encontrar pares en este intervalo
        mask = (dist_mat >= lag_min) & (dist_mat < lag_max)
        pairs = diff_sq[mask]
        
        if len(pairs) > 0:
            gamma = np.mean(pairs) / 2  # Semivarianza
            lags.append(lag_center)
            gammas.append(gamma)
            counts.append(len(pairs))
    
    return {
        'lags': np.array(lags),
        'gamma': np.array(gammas),
        'counts': np.array(counts)
    }

def cross_validation_kriging(df, c0, c1, a, modelo='esferico'):
    """
    Validación cruzada leave-one-out para evaluar calidad del modelo.
    
    Returns:
        DataFrame con valores reales, estimados, errores y métricas
    """
    resultados = []
    n = len(df)
    
    for i in range(n):
        # Dejar uno fuera
        df_train = df.drop(i).reset_index(drop=True)
        target = df.iloc[i][['X', 'Y']].values
        real = df.iloc[i]['Ley']
        
        # Estimar usando los demás
        res = resolver_kriging(df_train, target, c0, c1, a)
        
        if res['status'] == 'OK':
            estimado = res['ley']
            varianza = res['var']
            error = real - estimado
            error_std = error / np.sqrt(varianza) if varianza > 0 else 0
            
            resultados.append({
                'ID': df.iloc[i]['Id'],
                'Real': real,
                'Estimado': estimado,
                'Error': error,
                'Error_Std': error_std,
                'Varianza_K': varianza
            })
    
    df_cv = pd.DataFrame(resultados)
    
    # Calcular métricas globales
    if len(df_cv) > 0:
        rmse = np.sqrt(np.mean(df_cv['Error']**2))
        mae = np.mean(np.abs(df_cv['Error']))
        me = np.mean(df_cv['Error'])  # Mean Error (sesgo)
        r2 = 1 - (np.sum(df_cv['Error']**2) / np.sum((df_cv['Real'] - df_cv['Real'].mean())**2))
        
        metricas = {
            'RMSE': rmse,
            'MAE': mae,
            'ME': me,
            'R2': r2,
            'n_samples': len(df_cv)
        }
    else:
        metricas = None
    
    return df_cv, metricas

def kriging_grid(df, c0, c1, a, grid_resolution=30):
    """
    Genera una superficie interpolada de toda el área usando Kriging.
    
    Args:
        df: DataFrame con datos
        c0, c1, a: Parámetros del variograma
        grid_resolution: Número de puntos en cada dirección
    
    Returns:
        dict con 'X', 'Y', 'Z' (ley estimada), 'Var' (varianza)
    """
    # Determinar límites del área
    x_min, x_max = df['X'].min(), df['X'].max()
    y_min, y_max = df['Y'].min(), df['Y'].max()
    
    # Agregar margen del 10%
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    # Crear grid
    x_grid = np.linspace(x_min - x_margin, x_max + x_margin, grid_resolution)
    y_grid = np.linspace(y_min - y_margin, y_max + y_margin, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Estimar cada punto del grid
    Z = np.zeros_like(X)
    Var = np.zeros_like(X)
    
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            target = [X[i, j], Y[i, j]]
            res = resolver_kriging(df, target, c0, c1, a)
            if res['status'] == 'OK':
                Z[i, j] = res['ley']
                Var[i, j] = res['var']
            else:
                Z[i, j] = np.nan
                Var[i, j] = np.nan
    
    return {
        'X': X,
        'Y': Y,
        'Z': Z,
        'Var': Var
    }

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
    st.success("Sistema en Línea")
    st.markdown("<div style='text-align:center; color:#555; font-size:0.8em;'>v2.0 Build 2025</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#6b7280; font-size:0.75em; margin-top:10px;'>Desarrollado por<br><b style='color:#90caf9;'>Felix Bautista Layme</b></div>", unsafe_allow_html=True)

# --- CABECERA PRINCIPAL ---
st.title(f"{st.session_state['project_name']}")
st.markdown(f"#### Simulador de Estimación de Recursos Minerales con Kriging | Curso de Geoestadística Minera")

# Definición de Pestañas (Nombres cortos para que se vean todos en pantalla)
tabs = st.tabs([
    "1. Datos", 
    "2. Variograma", 
    "3. Kriging", 
    "4. Cálculos",
    "5. Economía",
    "6. JORC", 
    "7. Informe"
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
        st.subheader("📊 Análisis Estadístico Profesional")
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
            
            # Detección de Outliers (Método IQR)
            Q1 = df_calc['Ley'].quantile(0.25)
            Q3 = df_calc['Ley'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_calc[(df_calc['Ley'] < lower_bound) | (df_calc['Ley'] > upper_bound)]
            n_outliers = len(outliers)
            
            # --- TARJETAS MÉTRICAS PRINCIPALES ---
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("📊 Media (Ley)", f"{media:.3f} %", help="Promedio aritmético de las leyes")
            col_m2.metric("📏 Desv. Std.", f"{std:.3f}", help="Dispersión de los datos respecto a la media")
            col_m3.metric("📈 Coef. Variación", f"{cv:.1f} %", 
                         delta="Alto Riesgo" if cv>100 else "Estable", 
                         delta_color="inverse",
                         help="CV = (Std/Media)*100. Indica variabilidad relativa")
            col_m4.metric("⚠️ Outliers", f"{n_outliers}", 
                         delta=f"{n_outliers/len(df_calc)*100:.1f}%" if n_outliers > 0 else "0%",
                         delta_color="inverse",
                         help="Valores atípicos detectados (método IQR)")
            
            st.markdown("---")
            
            # --- TABLA DE PERCENTILES ---
            col_p1, col_p2 = st.columns([1, 1])
            
            with col_p1:
                st.markdown("**📊 Distribución por Percentiles:**")
                percentiles_df = pd.DataFrame({
                    'Percentil': ['P10', 'P25 (Q1)', 'P50 (Mediana)', 'P75 (Q3)', 'P90', 'P95'],
                    'Ley (%)': [
                        df_calc['Ley'].quantile(0.10),
                        Q1,
                        mediana,
                        Q3,
                        df_calc['Ley'].quantile(0.90),
                        df_calc['Ley'].quantile(0.95)
                    ]
                })
                st.dataframe(percentiles_df.style.format({'Ley (%)': '{:.3f}'}), use_container_width=True)
            
            with col_p2:
                st.markdown("**🔍 Métricas de Calidad de Datos:**")
                calidad_df = pd.DataFrame({
                    'Métrica': ['Rango (Max-Min)', 'IQR (Q3-Q1)', 'Sesgo (Skewness)', 'Curtosis'],
                    'Valor': [
                        f"{max_val - min_val:.3f}",
                        f"{IQR:.3f}",
                        f"{skewness:.3f}",
                        f"{kurt:.3f}"
                    ]
                })
                st.dataframe(calidad_df, use_container_width=True)
            
            # --- INTERPRETACIÓN DOCENTE MEJORADA ---
            st.markdown(f"""
            <div class='math-step'>
                <b>🧠 Interpretación Geoestadística Profesional:</b><br>
                <ul>
                    <li><b>Coeficiente de Variación (CV = {cv:.2f}%):</b> 
                        {
                            "Excelente. Datos muy homogéneos, ideales para Kriging." if cv < 30 else 
                            "Aceptable. Variabilidad moderada típica de depósitos minerales." if cv < 50 else
                            "Alta variabilidad. Considerar análisis de poblaciones o capping de valores extremos." if cv < 100 else
                            "Muy alta variabilidad. Presencia de 'pepitas' o valores extremos. Revisar protocolo de muestreo."
                        }
                    </li>
                    <li><b>Sesgo (Skewness = {skewness:.2f}):</b> 
                        {
                            "Distribución simétrica (ideal para Kriging)." if abs(skewness) < 0.5 else
                            "Sesgo positivo moderado (cola derecha). Común en leyes minerales." if skewness > 0 else
                            "Sesgo negativo (cola izquierda). Verificar datos."
                        }
                    </li>
                    <li><b>Curtosis ({kurt:.2f}):</b> 
                        {
                            "Distribución normal (mesocúrtica)." if abs(kurt) < 0.5 else
                            "Distribución puntiaguda (leptocúrtica). Concentración en torno a la media." if kurt > 0 else
                            "Distribución aplanada (platicúrtica). Datos dispersos."
                        }
                    </li>
                    <li><b>Outliers ({n_outliers} detectados):</b> 
                        {
                            "No se detectaron valores atípicos. Excelente calidad de datos." if n_outliers == 0 else
                            f"Se detectaron {n_outliers} valores fuera del rango [Q1-1.5*IQR, Q3+1.5*IQR]. Revisar si son errores o valores reales de alta ley."
                        }
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # --- FILTRO INTERACTIVO DE LEY DE CORTE ---
            st.markdown("---")
            st.markdown("### Filtro Interactivo de Ley de Corte (Cut-Off)")
            cutoff_value = st.slider(
                "Ajuste la ley de corte mínima (%)",
                min_value=0.0,
                max_value=float(max_val),
                value=0.0,
                step=0.01,
                help="Los puntos con ley menor a este valor se mostrarán en gris"
            )
            
            # Crear columna de categoría para el filtro
            df_calc['Categoria'] = df_calc['Ley'].apply(lambda x: 'Mineral' if x >= cutoff_value else 'Estéril')
            
            # Estadísticas del filtro
            n_mineral = len(df_calc[df_calc['Categoria'] == 'Mineral'])
            n_esteril = len(df_calc[df_calc['Categoria'] == 'Estéril'])
            
            col_f1, col_f2, col_f3 = st.columns(3)
            col_f1.metric("Mineral", f"{n_mineral}", f"{n_mineral/len(df_calc)*100:.1f}%")
            col_f2.metric("Estéril", f"{n_esteril}", f"{n_esteril/len(df_calc)*100:.1f}%")
            col_f3.metric("Ley Promedio Mineral", f"{df_calc[df_calc['Categoria']=='Mineral']['Ley'].mean():.3f}%" if n_mineral > 0 else "N/A")
            
            # --- PESTAÑAS GRÁFICAS MEJORADAS ---
            t1, t2, t3, t4, t5 = st.tabs(["Distribución", "Mapa 2D", "Mapa 3D", "Derivas", "Outliers"])
            
            with t1:
                # Histograma con filtro de corte
                fig_dist = px.histogram(
                    df_calc, x="Ley", nbins=20, marginal="box", 
                    title="Distribución de Frecuencias de Ley",
                    color='Categoria',
                    color_discrete_map={'Mineral': '#00bcd4', 'Estéril': '#6b7280'},
                    hover_data=df_calc.columns
                )
                fig_dist.add_vline(x=media, line_dash="dash", line_color="red", annotation_text=f"Media={media:.2f}")
                fig_dist.add_vline(x=mediana, line_dash="dash", line_color="green", annotation_text=f"Mediana={mediana:.2f}")
                fig_dist.add_vline(x=cutoff_value, line_dash="dot", line_color="yellow", annotation_text=f"Cut-Off={cutoff_value:.2f}")
                fig_dist.update_layout(template="plotly_dark", height=400, bargap=0.1)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with t2:
                # Mapa 2D con filtro de corte
                fig_map = px.scatter(
                    df_calc, x='X', y='Y', size='Ley', color='Categoria',
                    hover_name='Id', title="Mapa de Ubicación de Sondajes (Vista en Planta)",
                    color_discrete_map={'Mineral': '#00bcd4', 'Estéril': '#6b7280'},
                    size_max=50,
                    labels={'X': 'Coordenada Este (m)', 'Y': 'Coordenada Norte (m)'},
                    hover_data=['Ley']
                )
                fig_map.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig_map, use_container_width=True)
            
            with t3:
                # NUEVO: Mapa 3D de Sondajes
                st.markdown("#### Visualización 3D de Sondajes (Rotación Interactiva)")
                st.info("Usa el mouse para rotar, zoom y explorar el modelo 3D. El eje Z representa la Ley (%).")
                
                fig_3d = px.scatter_3d(
                    df_calc, x='X', y='Y', z='Ley',
                    color='Categoria',
                    color_discrete_map={'Mineral': '#00bcd4', 'Estéril': '#6b7280'},
                    size='Ley',
                    hover_name='Id',
                    title="Modelo 3D de Distribución de Leyes",
                    labels={
                        'X': 'Coordenada Este (m)',
                        'Y': 'Coordenada Norte (m)',
                        'Ley': 'Ley (%)'
                    },
                    size_max=15,
                    opacity=0.8
                )
                
                # Mejorar el diseño 3D
                fig_3d.update_layout(
                    template="plotly_dark",
                    height=600,
                    scene=dict(
                        xaxis_title='Este (m)',
                        yaxis_title='Norte (m)',
                        zaxis_title='Ley (%)',
                        bgcolor='#0e1117',
                        xaxis=dict(gridcolor='#374151', showbackground=True, backgroundcolor='#1a1a1a'),
                        yaxis=dict(gridcolor='#374151', showbackground=True, backgroundcolor='#1a1a1a'),
                        zaxis=dict(gridcolor='#374151', showbackground=True, backgroundcolor='#1a1a1a'),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.3)
                        )
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(0,0,0,0.5)"
                    )
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Estadísticas 3D
                col_3d1, col_3d2, col_3d3 = st.columns(3)
                col_3d1.metric("📏 Rango Este-Oeste", f"{df_calc['X'].max() - df_calc['X'].min():.1f} m")
                col_3d2.metric("📏 Rango Norte-Sur", f"{df_calc['Y'].max() - df_calc['Y'].min():.1f} m")
                col_3d3.metric("📏 Rango de Leyes", f"{min_val:.3f} - {max_val:.3f} %")

            with t4:
                # Análisis de Deriva (Drift Analysis)
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    fig_dx = px.scatter(df_calc, x='X', y='Ley', trendline="ols", 
                                       title="Deriva Este-Oeste", trendline_color_override="red")
                    fig_dx.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig_dx, use_container_width=True)
                with c_d2:
                    fig_dy = px.scatter(df_calc, x='Y', y='Ley', trendline="ols", 
                                       title="Deriva Norte-Sur", trendline_color_override="red")
                    fig_dy.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig_dy, use_container_width=True)
                
                st.info("💡 Si las líneas de tendencia tienen pendiente significativa, existe deriva espacial que debe considerarse.")
            
            with t5:
                # Visualización de Outliers
                if n_outliers > 0:
                    fig_outliers = go.Figure()
                    
                    # Datos normales
                    df_normal = df_calc[(df_calc['Ley'] >= lower_bound) & (df_calc['Ley'] <= upper_bound)]
                    fig_outliers.add_trace(go.Scatter(
                        x=df_normal['X'], y=df_normal['Y'],
                        mode='markers',
                        name='Datos Normales',
                        marker=dict(size=10, color='#4caf50', opacity=0.6)
                    ))
                    
                    # Outliers
                    fig_outliers.add_trace(go.Scatter(
                        x=outliers['X'], y=outliers['Y'],
                        mode='markers+text',
                        name='Outliers',
                        text=outliers['Id'],
                        textposition="top center",
                        marker=dict(size=15, color='#ff1744', symbol='star', line=dict(color='white', width=2))
                    ))
                    
                    fig_outliers.update_layout(
                        title=f"Ubicación Espacial de Outliers ({n_outliers} detectados)",
                        xaxis_title="Coordenada Este (m)",
                        yaxis_title="Coordenada Norte (m)",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_outliers, use_container_width=True)
                    
                    # Tabla de outliers
                    st.markdown("**📋 Detalle de Valores Atípicos:**")
                    st.dataframe(outliers[['Id', 'X', 'Y', 'Ley']].style.format({'X': '{:.2f}', 'Y': '{:.2f}', 'Ley': '{:.3f}'}))
                    
                    st.warning(f"⚠️ Límites de detección: Inferior = {lower_bound:.3f}%, Superior = {upper_bound:.3f}%")
                else:
                    st.success("✅ No se detectaron valores atípicos en los datos. Excelente calidad de muestreo.")
        else:
            st.warning("⚠️ No hay datos válidos para procesar.")

# ==============================================================================
# TAB 2: VARIOGRAFÍA ESTRUCTURAL AVANZADA
# ==============================================================================
with tabs[1]:
    st.markdown("""
    <div class='theory-box'>
        <span class='theory-title'>📈 Módulo 2: Análisis Variográfico Profesional</span>
        <p>El variograma es el corazón de la Geoestadística. Aquí calculamos el <b>variograma experimental</b> de tus datos reales 
        y lo ajustamos con un <b>modelo teórico</b>. Este modelo captura la continuidad espacial del yacimiento y es fundamental 
        para el Kriging. En minería profesional, este paso requiere validación por un Geólogo Competente (JORC).</p>
    </div>
    """, unsafe_allow_html=True)
    
    cv1, cv2 = st.columns([1, 2.5])
    
    with cv1:
        st.subheader("🛠️ Configuración del Modelo")
        
        # Selector de modelo variográfico
        st.markdown("**Tipo de Modelo:**")
        modelo_tipo = st.selectbox(
            "Seleccione el modelo teórico:",
            ["Esférico", "Gaussiano", "Exponencial", "Potencial"],
            help="Cada modelo tiene características distintas según la geología"
        )
        
        st.markdown("---")
        st.markdown("**Parámetros del Modelo:**")
        
        v_c0 = st.number_input("1️⃣ Efecto Pepita (C0 - Nugget)", 0.0, 50.0, 0.015, step=0.001, format="%.3f", 
                               help="Variabilidad a distancia cero. Representa error de muestreo + microvariabilidad.")
        v_c1 = st.number_input("2️⃣ Meseta Parcial (C1 - Sill)", 0.0, 100.0, 0.085, step=0.001, format="%.3f", 
                               help="Varianza espacialmente estructurada.")
        v_a  = st.number_input("3️⃣ Rango / Alcance (a)", 1.0, 2000.0, 120.0, step=10.0, format="%.1f", 
                               help="Distancia donde se pierde la correlación espacial.")
        
        meseta_total = v_c0 + v_c1
        nugget_ratio = (v_c0 / meseta_total * 100) if meseta_total > 0 else 0
        
        st.info(f"🔢 **Meseta Total:** {meseta_total:.3f}")
        st.info(f"📊 **Ratio Nugget/Sill:** {nugget_ratio:.1f}%")
        
        # Interpretación automática
        if nugget_ratio < 20:
            st.success("✅ Excelente continuidad espacial")
        elif nugget_ratio < 40:
            st.warning("⚠️ Continuidad moderada")
        else:
            st.error("❌ Baja continuidad - Revisar muestreo")
        
        st.markdown("---")
        
        # Calcular variograma experimental automáticamente
        if st.button("🔬 Calcular Variograma Experimental", help="Calcula el variograma de tus datos reales"):
            with st.spinner("Calculando pares de puntos..."):
                vario_exp = calcular_variograma_experimental(df_calc, n_lags=12)
                st.session_state['vario_exp'] = vario_exp
                st.success(f"✅ Calculado con {len(vario_exp['lags'])} intervalos")
        
        # Botón para recalcular si ya existe
        if 'vario_exp' in st.session_state:
            if st.button("🔄 Recalcular Variograma Experimental"):
                with st.spinner("Recalculando..."):
                    vario_exp = calcular_variograma_experimental(df_calc, n_lags=12)
                    st.session_state['vario_exp'] = vario_exp
                    st.success(f"✅ Recalculado con {len(vario_exp['lags'])} intervalos")
        
        # Guía profesional
        with st.expander("📚 Guía de Interpretación Profesional"):
            st.markdown(f"""
            **Modelo {modelo_tipo}:**
            
            {
                "**Esférico**: Modelo más común en minería. Alcanza la meseta exactamente en el rango. Ideal para depósitos con límites definidos." if modelo_tipo == "Esférico" else
                "**Gaussiano**: Muy suave cerca del origen. Indica alta continuidad. Común en depósitos sedimentarios estratificados." if modelo_tipo == "Gaussiano" else
                "**Exponencial**: Alcanza la meseta asintóticamente. Útil para depósitos con transiciones graduales." if modelo_tipo == "Exponencial" else
                "**Potencial**: Sin meseta definida. Indica deriva o tendencia regional. Requiere análisis especial."
            }
            
            **Criterios de Calidad:**
            - **Nugget/Sill < 20%**: Datos de alta calidad
            - **Nugget/Sill 20-40%**: Calidad aceptable
            - **Nugget/Sill > 40%**: Revisar protocolo de muestreo
            
            **Rango Típico por Tipo de Depósito:**
            - Vetas: 20-100 m
            - Pórfidos: 100-500 m
            - Sedimentarios: 200-1000 m
            """)
    
    with cv2:
        # Seleccionar función de variograma según modelo
        if modelo_tipo == "Esférico":
            func_variograma = variograma_esferico
            color_modelo = '#00bcd4'
        elif modelo_tipo == "Gaussiano":
            func_variograma = variograma_gaussiano
            color_modelo = '#4caf50'
        elif modelo_tipo == "Exponencial":
            func_variograma = variograma_exponencial
            color_modelo = '#ff9800'
        else:  # Potencial
            func_variograma = variograma_potencial
            color_modelo = '#e91e63'
        
        # Generación de datos para el gráfico teórico
        h = np.linspace(0, v_a * 1.8, 150)
        gamma = func_variograma(h, v_c0, v_c1, v_a)
        
        fig_var = go.Figure()
        
        # Variograma Experimental (si existe)
        if 'vario_exp' in st.session_state:
            vexp = st.session_state['vario_exp']
            fig_var.add_trace(go.Scatter(
                x=vexp['lags'], 
                y=vexp['gamma'], 
                mode='markers',
                name='Variograma Experimental',
                marker=dict(
                    size=12,
                    color='#ff1744',
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                text=[f"Pares: {c}" for c in vexp['counts']],
                hovertemplate='<b>Distancia:</b> %{x:.1f} m<br><b>Gamma:</b> %{y:.4f}<br>%{text}<extra></extra>'
            ))
        
        # Curva del Modelo Teórico
        fig_var.add_trace(go.Scatter(
            x=h, 
            y=gamma, 
            mode='lines', 
            name=f'Modelo {modelo_tipo}', 
            line=dict(color=color_modelo, width=4)
        ))
        
        # Líneas de Referencia
        fig_var.add_hline(
            y=meseta_total, 
            line_dash="dash", 
            line_color="green", 
            annotation_text=f"Meseta = {meseta_total:.3f}", 
            annotation_position="top right"
        )
        fig_var.add_vline(
            x=v_a, 
            line_dash="dash", 
            line_color="orange", 
            annotation_text=f"Rango = {v_a:.1f} m", 
            annotation_position="bottom right"
        )
        
        # Anotación Nugget
        if v_c0 > 0:
            fig_var.add_annotation(
                x=v_a * 0.1, 
                y=v_c0, 
                text=f"Nugget = {v_c0:.3f}", 
                showarrow=True, 
                arrowhead=2, 
                ax=50, 
                ay=-40, 
                font=dict(color="yellow", size=12),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor="yellow"
            )
        
        fig_var.update_layout(
            title=f"<b>Análisis Variográfico - Modelo {modelo_tipo}</b>",
            xaxis_title="Distancia de Separación (h) [metros]",
            yaxis_title="Semivarianza - γ(h)",
            template="plotly_dark",
            height=550,
            legend=dict(
                yanchor="top", 
                y=0.99, 
                xanchor="left", 
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_var, use_container_width=True)
        
        # Métricas de ajuste si hay variograma experimental
        if 'vario_exp' in st.session_state:
            vexp = st.session_state['vario_exp']
            gamma_teorico = func_variograma(vexp['lags'], v_c0, v_c1, v_a)
            
            # Calcular bondad de ajuste
            ss_res = np.sum((vexp['gamma'] - gamma_teorico)**2)
            ss_tot = np.sum((vexp['gamma'] - np.mean(vexp['gamma']))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("📊 R² de Ajuste", f"{r2:.3f}", help="Calidad del ajuste: >0.8 es excelente")
            col_m2.metric("🎯 Puntos Experimentales", len(vexp['lags']))
            col_m3.metric("📏 Rango/Dist.Max", f"{v_a / np.max(vexp['lags']):.2f}x")


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
        st.subheader("📍 Configuración de Estimación")
        
        # Selector de modo de estimación
        modo_estimacion = st.radio(
            "Seleccione el modo de estimación:",
            ["Punto Único", "Malla 2D (Heatmap)"],
            help="Punto Único: estima un bloque específico. Malla 2D: genera un mapa de calor de toda el área"
        )
        
        if modo_estimacion == "Punto Único":
            st.markdown("#### Coordenadas del Bloque")
            # Pre-cargar valores centrales de los datos
            default_x = df_calc['X'].mean()
            default_y = df_calc['Y'].mean()
            
            tx = st.number_input("Coordenada Este (X)", value=float(round(default_x, 0)))
            ty = st.number_input("Coordenada Norte (Y)", value=float(round(default_y, 0)))
            
            st.divider()
            
            if st.button("EJECUTAR KRIGING"):
                with st.spinner('Resolviendo sistema matricial...'):
                    res = resolver_kriging(df_calc, [tx, ty], v_c0, v_c1, v_a)
                    if res['status'] == 'OK':
                        # Añadimos metadatos al resultado
                        res.update({'tx': tx, 'ty': ty, 'c0': v_c0, 'c1': v_c1, 'a': v_a, 'fecha': datetime.now()})
                        st.session_state['resultado'] = res
                        st.session_state['modo_estimacion'] = 'punto'
                        guardar_historial(res)
                        st.success("¡Cálculo Exitoso!")
                    else:
                        st.error(res['msg'])
                        st.session_state['resultado'] = None
        
        else:  # Malla 2D
            st.markdown("#### Parámetros de la Malla")
            st.info("Se generará una cuadrícula de bloques y se estimará la ley en cada uno usando Kriging.")
            
            grid_size = st.slider(
                "Resolución de la malla (NxN bloques)",
                min_value=5,
                max_value=20,
                value=10,
                help="Mayor resolución = más detalle pero más tiempo de cálculo"
            )
            
            st.divider()
            
            if st.button("EJECUTAR KRIGING EN MALLA"):
                with st.spinner(f'Estimando {grid_size}x{grid_size} = {grid_size**2} bloques... Esto puede tomar unos segundos...'):
                    grid_result = kriging_grid(df_calc, v_c0, v_c1, v_a, grid_resolution=grid_size)
                    
                    # Guardar resultado en session_state
                    st.session_state['grid_result'] = grid_result
                    st.session_state['modo_estimacion'] = 'malla'
                    st.session_state['grid_size'] = grid_size
                    st.success(f"✅ Malla de {grid_size}x{grid_size} bloques estimada correctamente!")

    with c_der:
        # Visualización según modo de estimación
        if st.session_state.get('modo_estimacion') == 'malla' and 'grid_result' in st.session_state:
            # MODO MALLA 2D - HEATMAP
            st.markdown("### 🗺️ Mapa de Calor - Estimación de Malla 2D")
            
            grid_data = st.session_state['grid_result']
            grid_size = st.session_state.get('grid_size', 10)
            
            # Crear heatmap
            fig_heatmap = go.Figure()
            
            # Agregar el heatmap de leyes estimadas
            fig_heatmap.add_trace(go.Heatmap(
                x=grid_data['X'][0, :],
                y=grid_data['Y'][:, 0],
                z=grid_data['Z'],
                colorscale='Turbo',
                colorbar=dict(title=dict(text="Ley Estimada (%)", side="right")),
                hovertemplate='Este: %{x:.1f} m<br>Norte: %{y:.1f} m<br>Ley: %{z:.3f}%<extra></extra>',
                name='Ley Estimada'
            ))
            
            # Superponer ubicación de sondajes
            fig_heatmap.add_trace(go.Scatter(
                x=df_calc['X'],
                y=df_calc['Y'],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='white',
                    symbol='circle',
                    line=dict(color='black', width=2)
                ),
                text=df_calc['Id'],
                textposition="top center",
                textfont=dict(size=8, color='white'),
                name='Sondajes',
                hovertemplate='<b>%{text}</b><br>Este: %{x:.1f} m<br>Norte: %{y:.1f} m<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title=f"Mapa de Calor de Leyes Estimadas ({grid_size}x{grid_size} bloques)",
                xaxis_title="Coordenada Este (m)",
                yaxis_title="Coordenada Norte (m)",
                template="plotly_dark",
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(0,0,0,0.7)"
                )
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Estadísticas de la malla
            st.markdown("### 📊 Estadísticas de la Malla Estimada")
            
            # Filtrar valores válidos (no NaN)
            valid_grades = grid_data['Z'][~np.isnan(grid_data['Z'])]
            valid_vars = grid_data['Var'][~np.isnan(grid_data['Var'])]
            
            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            col_g1.metric("📊 Ley Media", f"{np.mean(valid_grades):.3f}%")
            col_g2.metric("📈 Ley Máxima", f"{np.max(valid_grades):.3f}%")
            col_g3.metric("📉 Ley Mínima", f"{np.min(valid_grades):.3f}%")
            col_g4.metric("🎯 Bloques Estimados", f"{len(valid_grades)}")
            
            # Mapa de varianza
            with st.expander("🔍 Ver Mapa de Varianza de Kriging (Incertidumbre)"):
                fig_var_map = go.Figure()
                
                fig_var_map.add_trace(go.Heatmap(
                    x=grid_data['X'][0, :],
                    y=grid_data['Y'][:, 0],
                    z=grid_data['Var'],
                    colorscale='Reds',
                    colorbar=dict(title=dict(text="Varianza Kriging", side="right")),
                    hovertemplate='Este: %{x:.1f} m<br>Norte: %{y:.1f} m<br>Varianza: %{z:.4f}<extra></extra>'
                ))
                
                # Superponer sondajes
                fig_var_map.add_trace(go.Scatter(
                    x=df_calc['X'],
                    y=df_calc['Y'],
                    mode='markers',
                    marker=dict(size=10, color='white', symbol='circle', line=dict(color='black', width=2)),
                    name='Sondajes'
                ))
                
                fig_var_map.update_layout(
                    title="Mapa de Incertidumbre (Varianza de Kriging)",
                    xaxis_title="Coordenada Este (m)",
                    yaxis_title="Coordenada Norte (m)",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig_var_map, use_container_width=True)
                st.info("💡 Zonas rojas = Mayor incertidumbre (lejos de sondajes). Zonas oscuras = Menor incertidumbre (cerca de sondajes).")
        
        elif st.session_state['resultado'] and st.session_state['resultado']['status']=='OK':
            # MODO PUNTO ÚNICO (ORIGINAL)
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
                                title=f"Plano de Estimación (Bloque en X:{res['tx']:.1f}, Y:{res['ty']:.1f})", 
                                color_continuous_scale='Viridis',
                                hover_data=[col_id]) # <--- Aquí usamos la columna detectada
            
            # Añadir el bloque como un marcador distinto
            fig_plan.add_trace(go.Scatter(
                x=[res['tx']], y=[res['ty']], mode='markers+text', 
                marker=dict(color='#ff1744', size=30, symbol='square', line=dict(color='white', width=2)), 
                name='BLOQUE A ESTIMAR', text=["BLOQUE"], textposition="top center",
                textfont=dict(size=14, color="white", family="Arial Black")
            ))
            
            # Añadir Radio de Influencia
            t = np.linspace(0, 2*np.pi, 100)
            fig_plan.add_trace(go.Scatter(
                x=res['tx']+v_a*np.cos(t), y=res['ty']+v_a*np.sin(t), 
                mode='lines', line=dict(dash='dash', color='white', width=1), 
                name='Radio de Influencia (Rango)'
            ))
            
            fig_plan.update_layout(template="plotly_dark", height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_plan, use_container_width=True)
        else:
            st.info("⚠️ Ejecute primero una estimación (Punto Único o Malla 2D) en el panel izquierdo.")

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
        
        # --- NUEVA SECCIÓN: VALIDACIÓN CRUZADA ---
        st.markdown("---")
        st.markdown("### 🔹 Paso 5: Validación del Modelo (Cross-Validation)")
        
        st.markdown("""
        <div class='theory-box'>
            <span class='theory-title'>🎯 Validación Cruzada Leave-One-Out</span>
            <p>Para evaluar la calidad del modelo variográfico, estimamos cada muestra usando todas las demás. 
            Comparamos el valor <b>real</b> vs el <b>estimado</b> para calcular métricas de error. 
            En minería profesional, esto es obligatorio antes de reportar recursos.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔬 Ejecutar Validación Cruzada", help="Puede tomar unos segundos con muchas muestras"):
            with st.spinner(f"Validando {len(df_calc)} muestras..."):
                df_cv, metricas = cross_validation_kriging(df_calc, v_c0, v_c1, v_a)
                st.session_state['cross_validation'] = {'df': df_cv, 'metricas': metricas}
                st.success("✅ Validación completada!")
        
        if 'cross_validation' in st.session_state:
            cv_data = st.session_state['cross_validation']
            df_cv = cv_data['df']
            metricas = cv_data['metricas']
            
            # Validar que se obtuvieron métricas
            if metricas is None or len(df_cv) == 0:
                st.error("❌ No se pudo completar la validación cruzada. Verifique que tiene suficientes datos y que el modelo variográfico es válido.")
            else:
                # Mostrar métricas principales
                col_cv1, col_cv2, col_cv3, col_cv4 = st.columns(4)
                col_cv1.metric("📊 R² (Correlación)", f"{metricas['R2']:.3f}", 
                              help="Cercano a 1 es excelente. >0.7 es aceptable")
                col_cv2.metric("📉 RMSE", f"{metricas['RMSE']:.4f}", 
                              help="Root Mean Square Error - Menor es mejor")
                col_cv3.metric("📏 MAE", f"{metricas['MAE']:.4f}", 
                              help="Mean Absolute Error")
                col_cv4.metric("⚖️ Sesgo (ME)", f"{metricas['ME']:.4f}", 
                              help="Cercano a 0 es ideal (sin sesgo)")
                
                # Gráficos de validación
                tab_cv1, tab_cv2, tab_cv3 = st.tabs(["📊 Real vs Estimado", "📈 Q-Q Plot", "📉 Distribución de Errores"])
                
                with tab_cv1:
                    # Scatter plot Real vs Estimado
                    fig_scatter = px.scatter(
                        df_cv, x='Real', y='Estimado',
                        title="Validación Cruzada: Ley Real vs Estimada",
                        labels={'Real': 'Ley Real (%)', 'Estimado': 'Ley Estimada (%)'},
                        hover_data=['ID', 'Error']
                    )
                    
                    # Línea 1:1 (ideal)
                    min_val = min(df_cv['Real'].min(), df_cv['Estimado'].min())
                    max_val = max(df_cv['Real'].max(), df_cv['Estimado'].max())
                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines', name='Línea 1:1 (Ideal)',
                        line=dict(color='red', dash='dash', width=2)
                    ))
                    
                    fig_scatter.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    st.info(f"""
                    **Interpretación:**
                    - Puntos cerca de la línea roja = Buena estimación
                    - R² = {metricas['R2']:.3f} → {"Excelente correlación" if metricas['R2'] > 0.8 else "Correlación aceptable" if metricas['R2'] > 0.6 else "Revisar modelo"}
                    """)
                
                with tab_cv2:
                    # Q-Q Plot para verificar normalidad de errores
                    from scipy.stats import probplot
                    qq_data = probplot(df_cv['Error'], dist="norm")
                    
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(
                        x=qq_data[0][0], y=qq_data[0][1],
                        mode='markers', name='Errores',
                        marker=dict(color='#00bcd4', size=8)
                    ))
                    fig_qq.add_trace(go.Scatter(
                        x=qq_data[0][0], y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1],
                        mode='lines', name='Distribución Normal Teórica',
                        line=dict(color='red', dash='dash')
                    ))
                    fig_qq.update_layout(
                        title="Q-Q Plot (Normalidad de Errores)",
                        xaxis_title="Cuantiles Teóricos",
                        yaxis_title="Cuantiles de Errores",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_qq, use_container_width=True)
                    
                    st.info("Si los puntos siguen la línea roja, los errores son normales (asunción del Kriging cumplida)")
                
                with tab_cv3:
                    # Histograma de errores
                    fig_hist = px.histogram(
                        df_cv, x='Error', nbins=20,
                        title="Distribución de Errores de Estimación",
                        labels={'Error': 'Error (Real - Estimado)'},
                        color_discrete_sequence=['#4caf50']
                    )
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Error = 0")
                    fig_hist.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Estadísticas de errores
                    col_e1, col_e2, col_e3 = st.columns(3)
                    col_e1.metric("Media de Errores", f"{df_cv['Error'].mean():.4f}")
                    col_e2.metric("Std de Errores", f"{df_cv['Error'].std():.4f}")
                    col_e3.metric("Max Error Abs", f"{df_cv['Error'].abs().max():.4f}")
    else:
        st.info("⚠️ Primero ejecute la estimación en la pestaña 3.")

# ==============================================================================
# TAB 5: ECONOMÍA MINERA AVANZADA
# ==============================================================================
with tabs[4]:
    if st.session_state['resultado']:
        res = st.session_state['resultado']
        st.markdown("""
        <div class='theory-box'>
            <span class='theory-title'>💰 Módulo 5: Valorización Económica y Análisis Financiero</span>
            <p>Transformamos la estimación geológica en <b>valor económico</b>. Calculamos el valor neto del bloque considerando 
            geometría, densidad, recuperación metalúrgica, precios de mercado y costos operativos. 
            Esta evaluación determina si el bloque es <b>mineral económico</b> o <b>estéril</b>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Inputs Económicos en diseño mejorado
        st.markdown("### 📋 Parámetros de Evaluación Económica")
        
        ce1, ce2, ce3 = st.columns(3)
        with ce1:
            st.markdown("**🔷 Geometría del Bloque**")
            dim_x = st.number_input("📏 Largo (m)", value=20.0, min_value=1.0, max_value=100.0, step=1.0)
            dim_y = st.number_input("📏 Ancho (m)", value=20.0, min_value=1.0, max_value=100.0, step=1.0)
            dim_z = st.number_input("📏 Alto (Banco) (m)", value=10.0, min_value=1.0, max_value=50.0, step=1.0)
        with ce2:
            st.markdown("**⚙️ Parámetros Físicos y Metalúrgicos**")
            densidad = st.number_input("⚖️ Densidad (t/m³)", value=2.60, min_value=1.0, max_value=5.0, step=0.01)
            recup = st.number_input("🔬 Recuperación Metalúrgica (%)", value=88.0, min_value=0.0, max_value=100.0, step=0.5)
            dilucion = st.number_input("📊 Dilución (%)", value=5.0, min_value=0.0, max_value=30.0, step=1.0, 
                                       help="Porcentaje de material estéril mezclado con mineral")
        with ce3:
            st.markdown("**💵 Parámetros de Mercado**")
            precio = st.number_input("💰 Precio del Metal (US$/lb)", value=4.15, min_value=0.1, max_value=20.0, step=0.05)
            costo_minado = st.number_input("⛏️ Costo Minado (US$/t)", value=25.0, min_value=1.0, max_value=200.0, step=1.0)
            costo_proceso = st.number_input("🏭 Costo Proceso (US$/t)", value=20.0, min_value=1.0, max_value=200.0, step=1.0)

        # Cálculos Económicos Detallados
        volumen = dim_x * dim_y * dim_z
        tonelaje_insitu = volumen * densidad
        tonelaje_diluido = tonelaje_insitu * (1 + dilucion/100)
        ley_diluida = res['ley'] / (1 + dilucion/100)
        
        fino_ton = tonelaje_diluido * (ley_diluida/100)
        fino_lbs = fino_ton * 2204.62
        fino_recuperado_lbs = fino_lbs * (recup/100)
        fino_recuperado_ton = fino_recuperado_lbs / 2204.62
        
        ingreso_bruto = fino_recuperado_lbs * precio
        costo_minado_total = tonelaje_diluido * costo_minado
        costo_proceso_total = tonelaje_diluido * costo_proceso
        costo_total_bloque = costo_minado_total + costo_proceso_total
        profit = ingreso_bruto - costo_total_bloque
        
        # Métricas adicionales
        nsr_por_ton = ingreso_bruto / tonelaje_diluido if tonelaje_diluido > 0 else 0
        costo_por_ton = costo_total_bloque / tonelaje_diluido if tonelaje_diluido > 0 else 0
        margen_porcentual = (profit / ingreso_bruto * 100) if ingreso_bruto > 0 else 0
        ley_corte = (costo_por_ton / (precio * 2204.62 * recup/100)) * 100 if precio > 0 else 0
        
        st.divider()
        
        # --- DASHBOARD DE RESULTADOS ---
        st.markdown("### 📊 Dashboard Económico del Bloque")
        
        # KPIs Principales en tarjetas grandes
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: #93c5fd; margin: 0; font-size: 0.9em;'>TONELAJE TOTAL</p>
                <h2 style='color: white; margin: 10px 0; font-size: 2em;'>{tonelaje_diluido:,.0f}</h2>
                <p style='color: #bfdbfe; margin: 0;'>toneladas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #7c2d12 0%, #9a3412 100%); padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: #fed7aa; margin: 0; font-size: 0.9em;'>METAL RECUPERABLE</p>
                <h2 style='color: white; margin: 10px 0; font-size: 2em;'>{fino_recuperado_ton:.2f}</h2>
                <p style='color: #fde68a; margin: 0;'>toneladas finas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi3:
            color_profit = "#065f46" if profit > 0 else "#7f1d1d"
            color_text = "#6ee7b7" if profit > 0 else "#fca5a5"
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color_profit} 0%, {color_profit}dd 100%); padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: {color_text}; margin: 0; font-size: 0.9em;'>BENEFICIO NETO</p>
                <h2 style='color: white; margin: 10px 0; font-size: 2em;'>${profit:,.0f}</h2>
                <p style='color: {color_text}; margin: 0;'>USD</p>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi4:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4c1d95 0%, #5b21b6 100%); padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: #ddd6fe; margin: 0; font-size: 0.9em;'>MARGEN</p>
                <h2 style='color: white; margin: 10px 0; font-size: 2em;'>{margen_porcentual:.1f}%</h2>
                <p style='color: #e9d5ff; margin: 0;'>del ingreso</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- VISUALIZACIONES AVANZADAS ---
        tab_eco1, tab_eco2, tab_eco3, tab_eco4 = st.tabs(["Bloque 3D", "Flujo de Caja", "Sensibilidad", "Resumen"])
        
        with tab_eco1:
            # Visualización 3D del bloque
            st.markdown("#### Representación 3D del Bloque Minero")
            
            # Crear figura 3D
            fig_3d = go.Figure()
            
            # Definir vértices del bloque
            x_coords = [0, dim_x, dim_x, 0, 0, dim_x, dim_x, 0]
            y_coords = [0, 0, dim_y, dim_y, 0, 0, dim_y, dim_y]
            z_coords = [0, 0, 0, 0, dim_z, dim_z, dim_z, dim_z]
            
            # Crear mesh 3D
            fig_3d.add_trace(go.Mesh3d(
                x=x_coords, y=y_coords, z=z_coords,
                i=[0, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[1, 2, 3, 4, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[2, 3, 4, 5, 6, 7, 1, 1, 5, 5, 7, 6],
                color='#00bcd4',
                opacity=0.7,
                name='Bloque Minero'
            ))
            
            # Añadir anotaciones
            fig_3d.add_trace(go.Scatter3d(
                x=[dim_x/2], y=[dim_y/2], z=[dim_z/2],
                mode='text',
                text=[f"Ley: {ley_diluida:.3f}%<br>Valor: ${profit:,.0f}"],
                textfont=dict(size=14, color='white'),
                showlegend=False
            ))
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title=f'Largo: {dim_x} m',
                    yaxis_title=f'Ancho: {dim_y} m',
                    zaxis_title=f'Alto: {dim_z} m',
                    bgcolor='#0e1117',
                    xaxis=dict(gridcolor='#374151'),
                    yaxis=dict(gridcolor='#374151'),
                    zaxis=dict(gridcolor='#374151')
                ),
                title=f"Bloque {dim_x}m x {dim_y}m x {dim_z}m = {volumen:,.0f} m³",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Tabla de especificaciones
            col_spec1, col_spec2 = st.columns(2)
            with col_spec1:
                st.markdown("**📐 Especificaciones Geométricas:**")
                specs_geo = pd.DataFrame({
                    'Parámetro': ['Volumen', 'Tonelaje In-Situ', 'Tonelaje Diluido', 'Densidad'],
                    'Valor': [
                        f"{volumen:,.0f} m³",
                        f"{tonelaje_insitu:,.0f} t",
                        f"{tonelaje_diluido:,.0f} t",
                        f"{densidad:.2f} t/m³"
                    ]
                })
                st.dataframe(specs_geo, use_container_width=True, hide_index=True)
            
            with col_spec2:
                st.markdown("**⚗️ Especificaciones Metalúrgicas:**")
                specs_metal = pd.DataFrame({
                    'Parámetro': ['Ley Original', 'Ley Diluida', 'Recuperación', 'Metal Recuperable'],
                    'Valor': [
                        f"{res['ley']:.3f} %",
                        f"{ley_diluida:.3f} %",
                        f"{recup:.1f} %",
                        f"{fino_recuperado_ton:.2f} t"
                    ]
                })
                st.dataframe(specs_metal, use_container_width=True, hide_index=True)
        
        with tab_eco2:
            # Gráfico de cascada (Waterfall) para flujo de caja
            st.markdown("#### Análisis de Flujo de Caja (Waterfall Chart)")
            
            fig_waterfall = go.Figure(go.Waterfall(
                name="Flujo de Caja",
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=["Ingresos<br>Brutos", "Costos<br>Minado", "Costos<br>Proceso", "Beneficio<br>Neto"],
                y=[ingreso_bruto, -costo_minado_total, -costo_proceso_total, profit],
                text=[f"${ingreso_bruto:,.0f}", f"-${costo_minado_total:,.0f}", 
                      f"-${costo_proceso_total:,.0f}", f"${profit:,.0f}"],
                textposition="outside",
                connector={"line": {"color": "#6b7280"}},
                increasing={"marker": {"color": "#10b981"}},
                decreasing={"marker": {"color": "#ef4444"}},
                totals={"marker": {"color": "#3b82f6" if profit > 0 else "#dc2626"}}
            ))
            
            fig_waterfall.update_layout(
                title="Desglose del Valor Económico del Bloque",
                yaxis_title="Valor (US$)",
                template="plotly_dark",
                height=450,
                showlegend=False
            )
            
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
            # Métricas detalladas
            col_det1, col_det2, col_det3 = st.columns(3)
            col_det1.metric("💵 NSR por Tonelada", f"${nsr_por_ton:.2f}/t")
            col_det2.metric("💸 Costo por Tonelada", f"${costo_por_ton:.2f}/t")
            col_det3.metric("⚖️ Ley de Corte", f"{ley_corte:.3f}%", 
                           delta=f"{'Mineral' if ley_diluida > ley_corte else 'Estéril'}")
        
        with tab_eco3:
            # Análisis de sensibilidad
            st.markdown("#### Análisis de Sensibilidad - Tornado Diagram")
            
            # Variaciones de ±20%
            var_percent = 20
            
            # Calcular impactos
            sensibilidad = []
            
            # Precio
            profit_precio_alto = (fino_recuperado_lbs * precio * (1 + var_percent/100)) - costo_total_bloque
            profit_precio_bajo = (fino_recuperado_lbs * precio * (1 - var_percent/100)) - costo_total_bloque
            sensibilidad.append({
                'Variable': 'Precio Metal',
                'Bajo': profit_precio_bajo - profit,
                'Alto': profit_precio_alto - profit
            })
            
            # Ley
            ley_alta = ley_diluida * (1 + var_percent/100)
            ley_baja = ley_diluida * (1 - var_percent/100)
            profit_ley_alta = ((tonelaje_diluido * ley_alta/100) * 2204.62 * recup/100 * precio) - costo_total_bloque
            profit_ley_baja = ((tonelaje_diluido * ley_baja/100) * 2204.62 * recup/100 * precio) - costo_total_bloque
            sensibilidad.append({
                'Variable': 'Ley Mineral',
                'Bajo': profit_ley_baja - profit,
                'Alto': profit_ley_alta - profit
            })
            
            # Costos
            costo_alto = costo_total_bloque * (1 + var_percent/100)
            costo_bajo = costo_total_bloque * (1 - var_percent/100)
            sensibilidad.append({
                'Variable': 'Costos Operativos',
                'Bajo': (ingreso_bruto - costo_bajo) - profit,
                'Alto': (ingreso_bruto - costo_alto) - profit
            })
            
            # Recuperación
            recup_alta = min(recup * (1 + var_percent/100), 100)
            recup_baja = recup * (1 - var_percent/100)
            profit_recup_alta = (fino_lbs * recup_alta/100 * precio) - costo_total_bloque
            profit_recup_baja = (fino_lbs * recup_baja/100 * precio) - costo_total_bloque
            sensibilidad.append({
                'Variable': 'Recuperación',
                'Bajo': profit_recup_baja - profit,
                'Alto': profit_recup_alta - profit
            })
            
            df_sens = pd.DataFrame(sensibilidad)
            df_sens = df_sens.sort_values('Alto', ascending=True)
            
            # Crear tornado diagram
            fig_tornado = go.Figure()
            
            fig_tornado.add_trace(go.Bar(
                y=df_sens['Variable'],
                x=df_sens['Bajo'],
                name=f'-{var_percent}%',
                orientation='h',
                marker=dict(color='#ef4444')
            ))
            
            fig_tornado.add_trace(go.Bar(
                y=df_sens['Variable'],
                x=df_sens['Alto'],
                name=f'+{var_percent}%',
                orientation='h',
                marker=dict(color='#10b981')
            ))
            
            fig_tornado.update_layout(
                title=f"Impacto de Variaciones de ±{var_percent}% en el Beneficio Neto",
                xaxis_title="Cambio en Beneficio (US$)",
                barmode='overlay',
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig_tornado, use_container_width=True)
            
            st.info(f"""
            **💡 Interpretación:**
            - Las barras más largas indican mayor sensibilidad
            - El **{'Precio del Metal' if abs(df_sens.iloc[-1]['Alto']) > abs(df_sens.iloc[0]['Alto']) else df_sens.iloc[-1]['Variable']}** es la variable más crítica
            - Variaciones de ±{var_percent}% pueden cambiar el beneficio entre ${profit + df_sens['Bajo'].min():,.0f} y ${profit + df_sens['Alto'].max():,.0f}
            """)
            
            # --- NUEVA TABLA DE SENSIBILIDAD 2D ---
            st.markdown("---")
            st.markdown("#### Tabla de Sensibilidad 2D: Precio vs Recuperación")
            st.info("Esta tabla muestra cómo varía el **Valor del Bloque (US$)** ante cambios simultáneos en el Precio del Cobre y la Recuperación Metalúrgica.")
            
            # Definir variaciones
            precio_vars = [-10, 0, 10]  # Porcentajes
            recup_vars = [-5, 0, 5]     # Porcentajes
            
            # Crear matriz de sensibilidad
            sens_matrix = []
            for p_var in precio_vars:
                row = []
                for r_var in recup_vars:
                    # Calcular nuevo precio y recuperación
                    nuevo_precio = precio * (1 + p_var/100)
                    nueva_recup = min(recup * (1 + r_var/100), 100)
                    
                    # Calcular nuevo beneficio
                    nuevo_ingreso = (fino_lbs * nueva_recup/100) * nuevo_precio
                    nuevo_beneficio = nuevo_ingreso - costo_total_bloque
                    
                    row.append(nuevo_beneficio)
                sens_matrix.append(row)
            
            # Crear DataFrame
            df_sens_2d = pd.DataFrame(
                sens_matrix,
                columns=[f"Recup {r_var:+d}%" for r_var in recup_vars],
                index=[f"Precio {p_var:+d}%" for p_var in precio_vars]
            )
            
            # Función para colorear celdas
            def color_cells(val):
                if val > profit * 1.1:
                    return 'background-color: #065f46; color: white; font-weight: bold'
                elif val > profit * 0.9:
                    return 'background-color: #047857; color: white'
                elif val > 0:
                    return 'background-color: #fbbf24; color: black'
                else:
                    return 'background-color: #7f1d1d; color: white; font-weight: bold'
            
            # Mostrar tabla con formato
            st.dataframe(
                df_sens_2d.style.applymap(color_cells).format("${:,.0f}"),
                use_container_width=True
            )
            
            # Leyenda de colores
            col_leg1, col_leg2, col_leg3, col_leg4 = st.columns(4)
            col_leg1.markdown("🟢 **Verde Oscuro**: Beneficio > +10%")
            col_leg2.markdown("🟢 **Verde**: Beneficio estable (±10%)")
            col_leg3.markdown("🟡 **Amarillo**: Beneficio positivo pero bajo")
            col_leg4.markdown("🔴 **Rojo**: Pérdida económica")
            
            # --- GRÁFICO DE ARAÑA (SPIDER PLOT) ---
            st.markdown("---")
            st.markdown("#### Gráfico de Araña - Impacto Relativo de Variables")
            
            # Calcular impacto relativo (normalizado)
            impactos = {
                'Precio Metal': abs(profit_precio_alto - profit_precio_bajo),
                'Ley Mineral': abs(profit_ley_alta - profit_ley_baja),
                'Costos Operativos': abs((ingreso_bruto - costo_bajo) - (ingreso_bruto - costo_alto)),
                'Recuperación': abs(profit_recup_alta - profit_recup_baja)
            }
            
            # Normalizar a 100
            max_impacto = max(impactos.values())
            impactos_norm = {k: (v/max_impacto)*100 for k, v in impactos.items()}
            
            # Crear spider plot
            categories = list(impactos_norm.keys())
            values = list(impactos_norm.values())
            
            fig_spider = go.Figure()
            
            fig_spider.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Cerrar el polígono
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(0, 188, 212, 0.3)',
                line=dict(color='#00bcd4', width=3),
                name='Impacto Relativo'
            ))
            
            fig_spider.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        ticksuffix='%',
                        gridcolor='#374151'
                    ),
                    bgcolor='#0e1117'
                ),
                showlegend=False,
                title="Sensibilidad Relativa de Variables Económicas",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig_spider, use_container_width=True)
            
            st.success(f"""
            **🎯 Conclusión del Análisis de Sensibilidad:**
            
            La variable más crítica es **{max(impactos, key=impactos.get)}** con un impacto de ${max(impactos.values()):,.0f} USD ante variaciones del ±20%.
            
            **Recomendación:** Enfocar esfuerzos de control y optimización en esta variable para minimizar riesgos económicos.
            """)
        
        with tab_eco4:
            # Resumen ejecutivo
            st.markdown("#### 📋 Resumen Ejecutivo de Valorización")
            
            # Decisión de minado
            decision_color = "#10b981" if profit > 0 else "#ef4444"
            decision_text = "✅ MINERAL ECONÓMICO" if profit > 0 else "❌ ESTÉRIL ECONÓMICO"
            decision_icon = "🟢" if profit > 0 else "🔴"
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {decision_color}22 0%, {decision_color}11 100%); 
                        border: 3px solid {decision_color}; padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;'>
                <h1 style='color: {decision_color}; margin: 0; font-size: 3em;'>{decision_icon}</h1>
                <h2 style='color: {decision_color}; margin: 10px 0;'>{decision_text}</h2>
                <p style='color: #9ca3af; margin: 5px 0;'>Ley Estimada: {ley_diluida:.3f}% | Ley de Corte: {ley_corte:.3f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tabla resumen completa
            st.markdown("**📊 Tabla Resumen Completa:**")
            
            resumen_data = {
                'Categoría': [
                    'GEOMETRÍA', '', '', '',
                    'GEOLOGÍA', '', '',
                    'METALURGIA', '', '',
                    'ECONOMÍA', '', '', '', ''
                ],
                'Parámetro': [
                    'Volumen', 'Tonelaje In-Situ', 'Tonelaje Diluido', 'Densidad',
                    'Ley Original', 'Ley Diluida', 'Dilución',
                    'Recuperación', 'Metal Contenido', 'Metal Recuperable',
                    'Ingresos Brutos', 'Costos Minado', 'Costos Proceso', 'Costo Total', 'BENEFICIO NETO'
                ],
                'Valor': [
                    f"{volumen:,.0f} m³", f"{tonelaje_insitu:,.0f} t", f"{tonelaje_diluido:,.0f} t", f"{densidad:.2f} t/m³",
                    f"{res['ley']:.3f} %", f"{ley_diluida:.3f} %", f"{dilucion:.1f} %",
                    f"{recup:.1f} %", f"{fino_ton:.2f} t", f"{fino_recuperado_ton:.2f} t",
                    f"${ingreso_bruto:,.2f}", f"${costo_minado_total:,.2f}", f"${costo_proceso_total:,.2f}", 
                    f"${costo_total_bloque:,.2f}", f"${profit:,.2f}"
                ]
            }
            
            df_resumen = pd.DataFrame(resumen_data)
            st.dataframe(df_resumen, use_container_width=True, hide_index=True)
            
            # Guardar en session_state para el informe
            st.session_state['economia'] = {
                'volumen': volumen,
                'tonelaje': tonelaje_diluido,
                'ley_diluida': ley_diluida,
                'metal_recuperable': fino_recuperado_ton,
                'ingreso_bruto': ingreso_bruto,
                'costo_total': costo_total_bloque,
                'beneficio': profit,
                'margen': margen_porcentual,
                'ley_corte': ley_corte,
                'decision': decision_text
            }
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
# TAB 7: INFORME PROFESIONAL A4
# ==============================================================================
with tabs[6]:
    if st.session_state.get('resultado') is None:
        st.info("⚠️ Aún no has realizado ninguna estimación. Ve a la Pestaña 3 y calcula.")
    else:
        res = st.session_state['resultado']
        df_safe = st.session_state['df_data']
        
        # Validar que los datos coincidan
        if len(df_safe) != len(res['d_vec']):
            st.warning("⚠️ **¡ATENCIÓN!** Has cargado nuevos datos pero no has actualizado el cálculo.")
            st.error(f"Datos actuales: {len(df_safe)} muestras | Cálculo guardado: {len(res['d_vec'])} muestras.")
            st.markdown("👉 **SOLUCIÓN:** Ve a la **Pestaña 3 (Estimación)** y haz clic de nuevo en **'🚀 EJECUTAR KRIGING'**.")
            st.stop()
        
        st.markdown("### 📄 Generador de Informe Técnico Profesional (Formato A4)")
        st.info("💡 Este informe está optimizado para impresión en formato A4 e incluye TODAS las secciones: Estimación, Economía, Validación y Clasificación JORC.")
        
        # Obtener datos
        proj_name = st.session_state.get('project_name', 'PROYECTO SIN NOMBRE')
        student_names = st.session_state.get('student_names', ['Equipo Técnico'])
        economia_data = st.session_state.get('economia', None)
        
        # Generar informe usando el módulo
        try:
            html = generar_informe_html(res, df_safe, economia_data, proj_name, student_names)
            
            # Renderizar
            st.components.v1.html(html, height=1000, scrolling=True)
            
            # Botones de descarga
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                b64 = base64.b64encode(html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="Informe_Tecnico_{proj_name.replace(" ", "_")}.html" style="text-decoration:none;">' \
                       f'<button style="background-color:#2e7d32; color:white; padding:15px 30px; border:none; border-radius:8px; cursor:pointer; font-weight:bold; font-size:16px; width:100%;">' \
                       f'📥 DESCARGAR INFORME HTML</button></a>'
                st.markdown(href, unsafe_allow_html=True)
                st.caption("Formato HTML - Abrir en navegador e imprimir como PDF")
            
            with col_btn2:
                st.info("""
                **💡 Cómo imprimir en A4:**
                1. Descarga el archivo HTML
                2. Ábrelo en tu navegador
                3. Ctrl+P (Imprimir)
                4. Selecciona "Guardar como PDF"
                5. Tamaño: A4
                """)
            
            # Advertencia si falta economía
            if economia_data is None:
                st.warning("⚠️ **Nota:** La sección de economía no está incluida porque no has completado el Tab 5 (Economía). Completa esa sección para un informe más completo.")
        
        except Exception as e:
            st.error(f"❌ Error al generar el informe: {str(e)}")
            st.info("Por favor, verifica que hayas completado todos los pasos previos (Datos, Variograma, Estimación).")
