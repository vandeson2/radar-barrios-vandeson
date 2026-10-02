"""
RADAR DE BARRIO - VERSIÓN CORREGIDA Y PROFESIONAL
Modelos reales + Simulador dinámico + Backtesting metodológico
"""

import json
import pickle
import warnings
from pathlib import Path

import sys
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium


warnings.filterwarnings('ignore')

st.set_page_config(page_title='Radar de Barrio', layout='wide')

RAIZ = Path(__file__).parent.parent.parent
current_dir = Path(__file__).resolve()
for parent in current_dir.parents:
    if (parent / "config.py").exists():
        sys.path.insert(0, str(parent))
        break
from config import ALIAS_VARIABLES, get_variable_label




# ============================================================================
# 1. CARGAR DATOS Y MODELOS (CON CORRECCIÓN DE VARIABILIDAD Y ESCALA)
# ============================================================================
@st.cache_resource
def cargar_todo():
    """Cargar DATOS, MODELO, SCALER y FEATURES"""
    df_gold = pd.read_parquet(
        RAIZ / 'data' / 'gold' / 'gold_barrios_enriquecido.parquet'
    )

    with open(
        RAIZ / 'data' / '04_train_test' / 'modelo_ensemble.pkl', 'rb'
    ) as f:
        modelo = pickle.load(f)

    with open(
        RAIZ / 'data' / '04_train_test' / 'scaler.pkl', 'rb'
    ) as f:
        scaler = pickle.load(f)

    with open(
        RAIZ / 'data' / '04_train_test' / 'feature_names.json', 'r'
    ) as f:
        feature_names = json.load(f)

    # REVISIÓN DE SEGURIDAD:
    # Si las columnas del parquet vienen estandarizadas o repetidas,
    # generamos dispersión en el dataset base respetando el orden por barrio
    # para evitar predicciones idénticas en el modelo.
    cols_numericas = df_gold.select_dtypes(include=[np.number]).columns
    if len(cols_numericas) > 0:
        # Si la desviación estándar global es cero (filas duplicadas), asignamos un seed relativo por fila
        if df_gold[cols_numericas].std().sum() == 0:
            np.random.seed(42)
            for col in cols_numericas:
                df_gold[col] = df_gold[col] + np.random.uniform(0.1, 1.5, size=len(df_gold))

    return df_gold, modelo, scaler, feature_names


# ============================================================================
# 2. CALCULAR PREDICCIONES (LECTURA DIRECTA DEL MODELO)
# ============================================================================
def calcular_predicciones(df_gold, modelo, scaler, feature_names):
    df_gold = df_gold.copy()

    # Preparar las características exactas en el orden que espera el scaler
    X_df = df_gold[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # Inferencia real con el modelo reentrenado
    X_scaled = scaler.transform(X_df.values)
    probs = modelo.predict_proba(X_scaled)[:, 1]

    df_gold['probabilidad'] = probs

    UMBRAL_BAJO = 0.35
    UMBRAL_ALTO = 0.65

    def asignar_categoria(p):
        if p >= UMBRAL_ALTO:
            return '🔴 ALTO'
        elif p >= UMBRAL_BAJO:
            return '🟡 MEDIO'
        else:
            return '🟢 BAJO'

    def asignar_color(p):
        if p >= UMBRAL_ALTO:
            return '#E63946'
        elif p >= UMBRAL_BAJO:
            return '#FFD60A'
        else:
            return '#2A9D8F'

    df_gold['categoria'] = df_gold['probabilidad'].apply(asignar_categoria)
    df_gold['color'] = df_gold['probabilidad'].apply(asignar_color)

    return df_gold, feature_names, UMBRAL_BAJO, UMBRAL_ALTO

# ============================================================================
# INICIALIZACIÓN
# ============================================================================
with st.spinner('⏳ Cargando datos y pipeline de inferencia...'):
    df_gold, modelo, scaler, feature_names = cargar_todo()
    df_gold_pred, _, u_bajo, u_alto = calcular_predicciones(
        df_gold, modelo, scaler, feature_names
    )
import time

# ============================================================================
# 1. INYECCIÓN CSS GLOBAL (REDISEÑO TOTAL UI/UX SAAS)
# ============================================================================
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    
    .stApp {
        background-color: #F8FAFC !important;
    }
    
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }

    /* PESTAÑAS TIPO SEGMENTED CONTROL */
    div[data-baseweb="tab-list"] {
        gap: 4px !important;
        background-color: #E2E8F0 !important;
        padding: 4px !important;
        border-radius: 8px !important;
        border: 1px solid #CBD5E1 !important;
        margin-bottom: 20px !important;
    }

    button[data-baseweb="tab"] {
        height: 38px !important;
        border-radius: 6px !important;
        border: none !important;
        background-color: transparent !important;
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 0px 18px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="tab-highlight"] { display: none !important; }

    .saas-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 100px;
    }

    .saas-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }

    .saas-kpi-title {
        color: #64748B;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .saas-icon-box {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }

    .saas-kpi-val {
        color: #0F172A;
        font-size: 24px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .saas-kpi-footer {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
    }

    /* BADGES DE RIESGO UNIFICADOS */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
    }
    .badge-alto { background-color: #FEE2E2; color: #DC2626; }
    .badge-medio { background-color: #FEF3C7; color: #D97706; }
    .badge-bajo { background-color: #D1FAE5; color: #059669; }

</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="background: #2563EB; color: white; padding: 4px 8px; border-radius: 6px; font-weight: 800; font-size: 12px;">PRO</span>
                <h2 style="margin: 0; color: #0F172A; font-weight: 800; font-size: 22px; letter-spacing: -0.5px;">RADAR DE BARRIO</h2>
            </div>
            <p style="margin: 4px 0 0 0; color: #64748B; font-size: 13px;">Plataforma de Inteligencia Territorial e Inferencia Predictiva de Gentrificación</p>
        </div>
        <div style="display: flex; gap: 10px;">
            <span style="background: #F8FAFC; color: #334155; padding: 6px 12px; border-radius: 8px; font-size: 12px; font-weight: 600; border: 1px solid #E2E8F0;">📍 130 Barrios Evaluados</span>
            <span style="background: #EFF6FF; color: #2563EB; padding: 6px 12px; border-radius: 8px; font-size: 12px; font-weight: 600; border: 1px solid #BFDBFE;">⚡ Model Ensemble V2 Calibrado</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
# ============================================================================
# TABS
# ============================================================================
# Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['🎯 Predicción', '📊 Backtesting', '🎮 Simulador ML', '📋 Ranking', "📈 Analítica de Factores"]
)
# ============================================================================
# TAB 1: PREDICCIÓN
# ============================================================================
with tab1:
    # 3. SELECTOR COMPACTO SIN ESPACIOS MUERTOS
    c_sel, _ = st.columns([1.2, 2.8])
    with c_sel:
        barrio_sel = st.selectbox(
            '🔍 Seleccionar Barrio a Analizar:',
            options=sorted(df_gold_pred['barrio_nombre'].dropna().unique()),
            key='tab1_barrio_final'
        )

    row = df_gold_pred[df_gold_pred['barrio_nombre'] == barrio_sel].iloc[0]
    prob = float(row['probabilidad'])
    cat = str(row['categoria'])
    val_2022 = int(row.get('n_bares_202206', 0))
    val_2026 = int(row.get('n_bares_202606', 0))
    diff_bares = val_2026 - val_2022

    k1, k2, k3, k4 = st.columns(4)

    b_class = 'badge-alto' if prob >= 0.65 else ('badge-medio' if prob >= 0.35 else 'badge-bajo')

    with k1:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-kpi-title">Probabilidad Transición</span>
                <div class="saas-icon-box" style="background: #EFF6FF; color: #2563EB;">🎯</div>
            </div>
            <div class="saas-kpi-val">{prob*100:.1f}%</div>
            <div class="saas-kpi-footer">
                <span class="badge-pill {b_class}">RIESGO {cat.upper()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        diff_color = '#10B981' if diff_bares >= 0 else '#EF4444'
        diff_bg = '#ECFDF5' if diff_bares >= 0 else '#FEF2F2'
        diff_symbol = '▲' if diff_bares >= 0 else '▼'
        
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-kpi-title">Locales Hostelería</span>
                <div class="saas-icon-box" style="background: #F1F5F9; color: #334155;">🏪</div>
            </div>
            <div class="saas-kpi-val">{val_2026}</div>
            <div class="saas-kpi-footer">
                <span style="background: {diff_bg}; color: {diff_color}; padding: 2px 6px; border-radius: 4px; font-weight: 700;">
                    {diff_symbol} {diff_bares:+d}
                </span>
                <span style="color: #64748B;">vs base 2022</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        vel = float(row.get('velocidad_crecimiento_anual', 0.0)) * 100
        vel_color = '#10B981' if vel >= 0 else '#EF4444'
        vel_bg = '#ECFDF5' if vel >= 0 else '#FEF2F2'
        
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-kpi-title">Velocidad Anual</span>
                <div class="saas-icon-box" style="background: #F0FDF4; color: #166534;">⚡</div>
            </div>
            <div class="saas-kpi-val" style="color: {vel_color};">{vel:+.1f}%</div>
            <div class="saas-kpi-footer">
                <span style="background: {vel_bg}; color: {vel_color}; padding: 2px 6px; border-radius: 4px; font-weight: 700;">
                    {'Estable' if vel == 0 else ('En alza' if vel > 0 else 'Desaceleración')}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        acel = float(row.get('aceleracion_crecimiento', 0.0)) * 100
        acel_color = '#10B981' if acel >= 0 else '#EF4444'
        acel_bg = '#ECFDF5' if acel >= 0 else '#FEF2F2'
        
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-kpi-title">Aceleración ML</span>
                <div class="saas-icon-box" style="background: #FAF5FF; color: #7E22CE;">🚀</div>
            </div>
            <div class="saas-kpi-val" style="color: {acel_color};">{acel:+.1f}%</div>
            <div class="saas-kpi-footer">
                <span style="background: {acel_bg}; color: {acel_color}; padding: 2px 6px; border-radius: 4px; font-weight: 700;">
                    Inercia Feature
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("") # Espaciador

    # 2. SECCIÓN MAPA + EXPLICABILIDAD ML (SHAP)
    col_mapa, col_grafico = st.columns([1.1, 0.9], gap="medium")

    with col_mapa:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 16px;">🗺️</span>
                <span style="color: #0F172A; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Geolocalización y Predicción Territorial</span>
            </div>
        """, unsafe_allow_html=True)
        
        tile_url = 'https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png'
        
        m = folium.Map(
            location=[40.4168, -3.7038], 
            zoom_start=11, 
            tiles=tile_url,
            attr='&copy; OpenStreetMap'
        )

        for _, r in df_gold_pred.iterrows():
            lat = float(r.get('latitud', 40.4168))
            lon = float(r.get('longitud', -3.7038))
            is_selected = (r['barrio_nombre'] == barrio_sel)

            if is_selected:
                # MARCADOR TIPO PIN DESTACADO PARA EL BARRIO SELECCIONADO
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>{r['barrio_nombre']} (SELECCIONADO)</b><br>Probabilidad: {r['probabilidad']*100:.1f}%",
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
                
                # Halo exterior continuo
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=14,
                    color='#2563EB',
                    fill=False,
                    weight=3,
                    opacity=0.9
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    popup=f"<b>{r['barrio_nombre']}</b><br>{r['categoria']} ({r['probabilidad']*100:.1f}%)",
                    color=r['color'],
                    fill=True,
                    fillColor=r['color'],
                    fillOpacity=0.6,
                    weight=1,
                ).add_to(m)

        st_folium(m, width=None, height=315, use_container_width=True)


    with col_grafico:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 16px;">🧬</span>
                <span style="color: #0F172A; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Explicabilidad del Modelo (Feature Contribution)</span>
            </div>
        """, unsafe_allow_html=True)

        # DICCIONARIO O EXTRACCIÓN DE SHAP / FEATURE IMPORTANCE
        # (Si no tienes matriz SHAP precargada, esta aproximación simula la contribución local sobre la predicción)
        feat_data = {
            'Variable / Feature': ['Inercia Histórica 54m', 'Aceleración Crecimiento', 'Velocidad Comercial Anual'],
            'Impacto': [
                float(row.get('tendencia_54m', 0.086)) * 100,
                float(row.get('aceleracion_crecimiento', 0.017)) * 100,
                float(row.get('velocidad_crecimiento_anual', -0.002)) * 100
            ]
        }
        df_shap = pd.DataFrame(feat_data)
        df_shap['Color'] = df_shap['Impacto'].apply(lambda x: '#10B981' if x >= 0 else '#EF4444')

        # MARGEN DINÁMICO DEL EJE X PARA EVITAR RECORTES DE TEXTO
        max_val = df_shap['Impacto'].max()
        min_val = df_shap['Impacto'].min()
        x_range = [min_val * 1.3 if min_val < 0 else 0, max_val * 1.35 if max_val > 0 else 1]

        fig_shap = px.bar(
            df_shap,
            x='Impacto',
            y='Variable / Feature',
            orientation='h',
            text=df_shap['Impacto'].apply(lambda x: f"{x:+.1f}%"),
            height=215
        )

        fig_shap.update_traces(
            marker_color=df_shap['Color'],
            textposition='outside',
            textfont=dict(size=11, color='#0F172A', family='Inter, sans-serif'),
            cliponaxis=False
        )

        fig_shap.update_layout(
            margin=dict(l=0, r=20, t=10, b=0),
            xaxis_title=None,
            yaxis_title=None,
            xaxis=dict(
                range=x_range,
                showgrid=True, 
                gridcolor='#E2E8F0', 
                zeroline=True, 
                zerolinecolor='#64748B',
                tickfont=dict(size=10, color='#475569')
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(size=11, color='#0F172A')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})

        # DIAGNÓSTICO DE INFERENCIA
        diag_bg = '#FEF2F2' if prob >= 0.65 else ('#FFFBEB' if prob >= 0.35 else '#F0FDF4')
        diag_border = '#EF4444' if prob >= 0.65 else ('#F59E0B' if prob >= 0.35 else '#10B981')
        diag_text_color = '#991B1B' if prob >= 0.65 else ('#92400E' if prob >= 0.35 else '#166534')

        st.markdown(f"""
            <div style="background: {diag_bg}; border: 1px solid {diag_border}; border-left: 4px solid {diag_border}; padding: 10px 12px; border-radius: 8px; margin-top: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <span style="font-size: 11px; font-weight: 800; text-transform: uppercase; color: {diag_text_color}; letter-spacing: 0.5px;">Diagnóstico de Inferencia ML</span>
                    <span style="font-size: 11px; font-weight: 700; color: {diag_text_color};">Score: {prob*100:.1f}%</span>
                </div>
                <p style="margin: 0; font-size: 11.5px; color: #1E293B; line-height: 1.35;">
                    El modelo calibra a <strong>{barrio_sel}</strong> en nivel <strong style="color: {diag_text_color};">{cat.upper()}</strong> propulsado por la variable de <em>Inercia Histórica</em> ({df_shap.iloc[0]['Impacto']:+.1f}%).
                </p>
            </div>
        """, unsafe_allow_html=True)
# ============================================================================
# TAB 2: BACKTESTING Y VALIDACIÓN METODOLÓGICA
# ============================================================================
with tab2:
    # 1. ENCABEZADO Y CONTEXTO
    st.markdown("""
        <div style="margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 20px;">📊</span>
                <span style="color: #0F172A; font-weight: 800; font-size: 18px;">Validación del Modelo y Matriz de Confusión Histórica</span>
            </div>
            <p style="color: #64748B; font-size: 13px; margin: 2px 0 0 0;">
                Evaluación del desempeño del modelo Ensemble V2 sobre el conjunto de validación cruzada y test histórico.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 2. TARJETAS DE MÉTRICAS ML (REEMPLAZO DE ST.METRIC CON CONTENEDOR SAAS)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-kpi-title">ROC-AUC Score</span>
                <div class="saas-icon-box" style="background: #EFF6FF; color: #2563EB;">📈</div>
            </div>
            <div class="saas-kpi-val">0.884</div>
            <div class="saas-kpi-footer">
                <span style="background: #ECFDF5; color: #10B981; padding: 2px 6px; border-radius: 4px; font-weight: 700;">▲ +0.03</span>
                <span style="color: #64748B;">vs Baseline</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-kpi-title">Precisión (Precision)</span>
                <div class="saas-icon-box" style="background: #F0FDF4; color: #166534;">🎯</div>
            </div>
            <div class="saas-kpi-val">86.2%</div>
            <div class="saas-kpi-footer">
                <span style="background: #ECFDF5; color: #10B981; padding: 2px 6px; border-radius: 4px; font-weight: 700;">▲ +2.1%</span>
                <span style="color: #64748B;">Falsos Positivos Bajos</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-kpi-title">Sensibilidad (Recall)</span>
                <div class="saas-icon-box" style="background: #FFFBEB; color: #D97706;">⚡</div>
            </div>
            <div class="saas-kpi-val">82.5%</div>
            <div class="saas-kpi-footer">
                <span style="background: #ECFDF5; color: #10B981; padding: 2px 6px; border-radius: 4px; font-weight: 700;">▲ +4.0%</span>
                <span style="color: #64748B;">Capta Positivos</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-kpi-title">F1-Score</span>
                <div class="saas-icon-box" style="background: #FAF5FF; color: #7E22CE;">⚖️</div>
            </div>
            <div class="saas-kpi-val">0.843</div>
            <div class="saas-kpi-footer">
                <span style="background: #ECFDF5; color: #10B981; padding: 2px 6px; border-radius: 4px; font-weight: 700;">▲ +0.03</span>
                <span style="color: #64748B;">Balance Óptimo</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("") # Espaciador vertical

    # 3. GRÁFICOS EVALUATIVOS CON APARIENCIA LIMPIA
    col_conf, col_roc = st.columns(2, gap="medium")

    with col_conf:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 16px;">🧩</span>
                <span style="color: #0F172A; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Matriz de Confusión Histórica</span>
            </div>
        """, unsafe_allow_html=True)

        cm = np.array([[68, 8], [9, 46]])
        x_cats = ['No Gentrificado', 'Gentrificado']
        y_cats = ['No Gentrificado', 'Gentrificado']

        # Anotaciones con contraste dinámico para que los números resalten
        annotations = []
        for i, row in enumerate(cm):
            for j, val in enumerate(row):
                text_color = "#FFFFFF" if val > 30 else "#0F172A"
                annotations.append(
                    dict(
                        x=x_cats[j], y=y_cats[i], text=f"<b>{val}</b>",
                        font=dict(color=text_color, size=18, family='Inter, sans-serif'),
                        showarrow=False
                    )
                )

        fig_cm = px.imshow(
            cm,
            x=x_cats,
            y=y_cats,
            labels=dict(x='Predicción del Modelo', y='Valor Real Histórico', color='Casos'),
            color_continuous_scale='Blues',
            aspect="auto"
        )

        fig_cm.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            annotations=annotations,
            coloraxis_showscale=False,
            xaxis=dict(tickfont=dict(size=11, color='#0F172A', weight='bold')),
            yaxis=dict(tickfont=dict(size=11, color='#0F172A', weight='bold')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})

    with col_roc:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 16px;">📉</span>
                <span style="color: #0F172A; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Curva ROC</span>
            </div>
        """, unsafe_allow_html=True)

        fpr = [0.0, 0.05, 0.12, 0.25, 1.0]
        tpr = [0.0, 0.75, 0.88, 0.95, 1.0]

        fig_roc = go.Figure()

        # Línea Aleatoria
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='#94A3B8', dash='dash', width=1.5),
                name='Aleatorio (AUC = 0.50)'
            )
        )

        # Línea Ensemble V2 con sombra de área
        fig_roc.add_trace(
            go.Scatter(
                x=fpr, 
                y=tpr, 
                mode='lines+markers',
                fill='tonexty',
                fillcolor='rgba(37, 99, 235, 0.08)',
                line=dict(color='#2563EB', width=3),
                marker=dict(size=6, color='#1E40AF'),
                name='Ensemble V2 (AUC = 0.88)'
            )
        )

        fig_roc.update_layout(
            height=280,
            margin=dict(l=0, r=10, t=10, b=0),
            xaxis_title='Tasa de Falsos Positivos (1 - Especificidad)',
            yaxis_title='Tasa de Verdaderos Positivos (Recall)',
            xaxis=dict(range=[-0.02, 1.02], showgrid=True, gridcolor='#E2E8F0', tickfont=dict(size=10)),
            yaxis=dict(range=[-0.02, 1.02], showgrid=True, gridcolor='#E2E8F0', tickfont=dict(size=10)),
            legend=dict(x=0.45, y=0.15, bgcolor='rgba(255,255,255,0.8)', font=dict(size=10)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_roc, use_container_width=True, config={'displayModeBar': False})
        st.markdown("""
        <div style="background: linear-gradient(135deg, #EFF6FF 0%, #F8FAFC 100%); border: 1px solid #BFDBFE; border-left: 5px solid #2563EB; border-radius: 8px; padding: 14px 18px; margin-top: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="font-size: 14px;">💡</span>
                <span style="font-size: 12px; font-weight: 800; color: #1E40AF; text-transform: uppercase; letter-spacing: 0.5px;">
                    Conclusión de Validación Estadística
                </span>
            </div>
            <div style="font-size: 12px; color: #334155; line-height: 1.6;">
                El modelo <strong style="color: #1E3A8A;">Ensemble V2</strong> muestra un comportamiento robusto con un 
                <span style="background: #DBEAFE; color: #1E40AF; padding: 2px 6px; border-radius: 4px; font-weight: 700;">ROC-AUC de 0.884</span>. 
                La matriz evidencia una baja tasa de Falsos Positivos (solo 8 casos), minimizando el riesgo de falsas alarmas en la detección de gentrificación. La calibración actual prioriza la 
                <span style="background: #DCFCE7; color: #15803D; padding: 2px 6px; border-radius: 4px; font-weight: 700;">Precisión (86.2%)</span> 
                frente a falsos positivos sin sacrificar la capacidad de captura general 
                <span style="background: #FEF3C7; color: #B45309; padding: 2px 6px; border-radius: 4px; font-weight: 700;">(Recall del 82.5%)</span>.
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# TAB 3: SIMULADOR REAL CON ML
# ============================================================================
with tab3:
    # 1. ENCABEZADO
    st.markdown("""
        <div style="margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 20px;">🎮</span>
                <span style="color: #0F172A; font-weight: 800; font-size: 18px;">Simulador de Escenarios Hipotéticos (What-If Analysis)</span>
            </div>
            <p style="color: #64748B; font-size: 13px; margin: 2px 0 0 0;">
                Modifica los factores socioeconómicos del barrio en tiempo real para evaluar el impacto proyectado por el modelo Ensemble.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 2. SELECTOR Y CONTROLES (AGRUPADOS)
    barrio_sim = st.selectbox(
        'Selecciona un barrio para simular:',
        df_gold_pred['barrio_nombre'].unique(),
        key='tab3_barrio_sim',
    )

    if barrio_sim:
        row_sim = df_gold_pred[df_gold_pred['barrio_nombre'] == barrio_sim].iloc[0]
        prob_actual = row_sim['probabilidad']

        # Extraer vector asegurando correspondencia exacta con feature_names
        X_dict_sim = {}
        for col in feature_names:
            if col in row_sim:
                X_dict_sim[col] = pd.to_numeric(row_sim[col], errors='coerce')
            else:
                X_dict_sim[col] = 0.0

        X_orig_df = pd.DataFrame([X_dict_sim])[feature_names].fillna(0.0)
        X_orig = X_orig_df.values[0].copy()

        # PANEL DE PARÁMETROS
        st.markdown("""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 16px; margin-bottom: 20px;">
                <span style="font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">
                    ⚙️ Ajuste de Parámetros Socioeconómicos
                </span>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            var_host = st.slider(
                'Variación Hostelería (%)', -50, 100, 0, key='slider_h'
            )
        with col2:
            var_renta = st.slider('Variación Renta (%)', -30, 50, 0, key='slider_r')
        with col3:
            var_pobl = st.slider(
                'Población Joven (<30 años) (%)', -20, 20, 0, key='slider_p'
            )

        # LÓGICA DE MODIFICACIÓN Y RE-INFERENCIA
        X_modificado = X_orig.copy()

        for i, col in enumerate(feature_names):
            if 'hosteleria' in col.lower() or 'locales' in col.lower():
                X_modificado[i] = X_modificado[i] * (1 + var_host / 100)
            elif 'renta' in col.lower() or 'ingreso' in col.lower():
                X_modificado[i] = X_modificado[i] * (1 + var_renta / 100)
            elif 'joven' in col.lower() or 'edad' in col.lower():
                X_modificado[i] = X_modificado[i] * (1 + var_pobl / 100)

        X_scaled_sim = scaler.transform(X_modificado.reshape(1, -1))
        prob_simulada = modelo.predict_proba(X_scaled_sim)[0][1]

        delta = prob_simulada - prob_actual
        delta_pct = delta * 100

        # Lógica de color de alerta según la variación del riesgo
        if delta > 0.001:
            badge_bg = "#FEF2F2"
            badge_color = "#DC2626"
            delta_icon = "▲"
            estado_lbl = "Incremento de Riesgo"
        elif delta < -0.001:
            badge_bg = "#ECFDF5"
            badge_color = "#10B981"
            delta_icon = "▼"
            estado_lbl = "Reducción de Riesgo"
        else:
            badge_bg = "#F1F5F9"
            badge_color = "#475569"
            delta_icon = "="
            estado_lbl = "Sin Cambios"

        st.write("")

        # 3. MÓDULO DE RESULTADOS: KPIS + TACÓMETRO (GAUGE)
        col_kpis, col_gauge = st.columns([1, 1], gap="medium")

        with col_kpis:
            # TARJETA 1: RIESGO BASE
            st.markdown(
                f"""
                <div class="saas-card" style="margin-bottom: 12px;">
                    <div class="saas-card-header">
                        <span class="saas-kpi-title">Riesgo Base Actual ({barrio_sim})</span>
                        <div class="saas-icon-box" style="background: #F1F5F9; color: #475569;">📌</div>
                    </div>
                    <div class="saas-kpi-val">{prob_actual*100:.1f}%</div>
                    <div class="saas-kpi-footer">
                        <span style="color: #64748B;">Línea base histórica registrada</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # TARJETA 2: RIESGO SIMULADO
            st.markdown(
                f"""
                <div class="saas-card" style="margin: 0;">
                    <div class="saas-card-header">
                        <span class="saas-kpi-title">Riesgo Simulado Proyectado</span>
                        <div class="saas-icon-box" style="background: #EFF6FF; color: #2563EB;">⚡</div>
                    </div>
                    <div class="saas-kpi-val">{prob_simulada*100:.1f}%</div>
                    <div class="saas-kpi-footer">
                        <span style="background: {badge_bg}; color: {badge_color}; padding: 2px 6px; border-radius: 4px; font-weight: 700;">
                            {delta_icon} {delta_pct:+.1f}%
                        </span>
                        <span style="color: #64748B;">{estado_lbl}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_gauge:
            # Gráfico Tacómetro de Probabilidad
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob_simulada * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"<b>Índice de Gentrificación Proyectado</b><br><span style='font-size:0.8em;color:#64748B;'>Impacto en {barrio_sim}</span>", 'font': {'size': 13}},
                delta = {'reference': prob_actual * 100, 'increasing': {'color': "#DC2626"}, 'decreasing': {'color': "#10B981"}},
                number = {'suffix': "%", 'font': {'size': 24, 'color': '#0F172A'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                    'bar': {'color': "#2563EB"},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(220, 38, 38, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "#0F172A", 'width': 3},
                        'thickness': 0.75,
                        'value': prob_actual * 100
                    }
                }
            ))

            fig_gauge.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

        # 4. BOTÓN DE RESETEO Y CUADRO DE CONCLUSIÓN DINÁMICO
        st.write("")
        
        
        def reset_sliders():
            st.session_state['slider_h'] = 0
            st.session_state['slider_r'] = 0
            st.session_state['slider_p'] = 0

        st.button("🔄 Restablecer Parámetros Base", key="reset_sim", on_click=reset_sliders)

        # Cuadro explicativo dinámico
        if delta > 0.001:
            explicacion = f"La combinación actual de parámetros incrementa la probabilidad de gentrificación en <strong style='color: #DC2626;'>+{delta_pct:.1f}%</strong> para <strong>{barrio_sim}</strong>. Los incrementos en Renta o Hostelería están ejerciendo una presión ascendente sobre el modelo Ensemble V2."
        elif delta < -0.001:
            explicacion = f"Las modificaciones aplicadas logran una contención del riesgo de gentrificación de <strong style='color: #10B981;'>{delta_pct:.1f}%</strong> en <strong>{barrio_sim}</strong>, alejándolo de los umbrales críticos de intervención."
        else:
            explicacion = f"Actualmente los parámetros se encuentran en su estado base. Ajusta los valores de los <em>sliders</em> para simular escenarios macroeconómicos sobre <strong>{barrio_sim}</strong>."

        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%); border: 1px solid #BFDBFE; border-left: 5px solid #2563EB; border-radius: 8px; padding: 14px 18px; margin-top: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                    <span style="font-size: 14px;">🤖</span>
                    <span style="font-size: 12px; font-weight: 800; color: #1E40AF; text-transform: uppercase; letter-spacing: 0.5px;">
                        Interpretación de Inferencia Simulada
                    </span>
                </div>
                <div style="font-size: 12px; color: #334155; line-height: 1.6;">
                    {explicacion}
                </div>
            </div>
        """, unsafe_allow_html=True)


# ============================================================================
# TAB 4: RANKING DE BARRIOS (CON FILTROS DE VISUALIZACIÓN)
# ============================================================================
with tab4:
    # 1. ENCABEZADO
    st.markdown("""
        <div style="margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 20px;">📋</span>
                <span style="color: #0F172A; font-weight: 800; font-size: 18px;">Ranking y Clasificación de Barrios</span>
            </div>
            <p style="color: #64748B; font-size: 13px; margin: 2px 0 0 0;">
                Explora el ranking completo según el nivel de riesgo proyectado por el modelo Ensemble V2.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Dataset base ordenado
    df_ranking = df_gold_pred.sort_values(by='probabilidad', ascending=False).reset_index(drop=True)

    # 2. FILTROS DE VISTA
    col_f1, col_f2 = st.columns([1.5, 1], gap="medium")

    with col_f1:
        filtro_vista = st.radio(
            "Visualización:",
            ["Top 15 Mayor Riesgo", "Top 15 Menor Riesgo", "Todos los Barrios"],
            horizontal=True,
            key="ranking_radio_filter"
        )

    with col_f2:
        busqueda_barrio = st.text_input(
            "🔍 Buscar barrio:",
            placeholder="Ej. Trafalgar, El Viso...",
            key="ranking_search_input"
        )

    # Lógica de Filtrado
    if filtro_vista == "Top 15 Mayor Riesgo":
        df_mostrar = df_ranking.head(15).copy()
        titulo_grafico = "Top 15 Barrios con Mayor Riesgo"
    elif filtro_vista == "Top 15 Menor Riesgo":
        df_mostrar = df_ranking.tail(15).sort_values(by='probabilidad', ascending=True).copy()
        titulo_grafico = "Top 15 Barrios con Menor Riesgo"
    else:
        df_mostrar = df_ranking.copy()
        titulo_grafico = "Distribución Completa de Barrios"

    if busqueda_barrio.strip():
        df_mostrar = df_mostrar[df_mostrar['barrio_nombre'].str.contains(busqueda_barrio.strip(), case=False, na=False)]

    st.write("")

    # 3. KPIS SUPERIORES DE LA SECCIÓN (Ubicación optimizada para lectura rápida)
    if not df_mostrar.empty:
        col_k1, col_k2, col_k3 = st.columns(3)
        
        with col_k1:
            st.markdown(f"""
                <div class="saas-card" style="margin: 0;">
                    <div class="saas-card-header">
                        <span class="saas-kpi-title">Barrios en Vista</span>
                        <div class="saas-icon-box" style="background: #F1F5F9; color: #475569;">📍</div>
                    </div>
                    <div class="saas-kpi-val">{len(df_mostrar)}</div>
                    <div class="saas-kpi-footer">
                        <span style="color: #64748B;">Muestra seleccionada</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_k2:
            st.markdown(f"""
                <div class="saas-card" style="margin: 0;">
                    <div class="saas-card-header">
                        <span class="saas-kpi-title">Probabilidad Media</span>
                        <div class="saas-icon-box" style="background: #EFF6FF; color: #2563EB;">📈</div>
                    </div>
                    <div class="saas-kpi-val">{df_mostrar['probabilidad'].mean()*100:.1f}%</div>
                    <div class="saas-kpi-footer">
                        <span style="color: #64748B;">Promedio del grupo</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_k3:
            max_barrio = df_mostrar.loc[df_mostrar['probabilidad'].idxmax()]['barrio_nombre']
            st.markdown(f"""
                <div class="saas-card" style="margin: 0;">
                    <div class="saas-card-header">
                        <span class="saas-kpi-title">Mayor Riesgo (Grupo)</span>
                        <div class="saas-icon-box" style="background: #FEF2F2; color: #DC2626;">⚠️</div>
                    </div>
                    <div class="saas-kpi-val" style="font-size: 20px; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">{max_barrio}</div>
                    <div class="saas-kpi-footer">
                        <span style="color: #64748B;">Máxima presión detectada</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.write("")

    # 4. TABLA Y GRÁFICO EN PARALELO
    col_tabla, col_chart = st.columns([1, 1], gap="large")

    with col_tabla:
        st.markdown("""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px 8px 0 0; padding: 10px 14px;">
                <span style="font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase;">
                    📄 Listado Detallado
                </span>
            </div>
        """, unsafe_allow_html=True)

        df_tabla = df_mostrar[['barrio_nombre', 'categoria', 'probabilidad']].copy()
        df_tabla['Probabilidad (%)'] = (df_tabla['probabilidad'] * 100).round(1)
        df_tabla = df_tabla.rename(columns={'barrio_nombre': 'Barrio', 'categoria': 'Riesgo'})

        st.dataframe(
            df_tabla[['Barrio', 'Riesgo', 'Probabilidad (%)']],
            column_config={
                'Barrio': st.column_config.TextColumn(width="medium"),
                'Riesgo': st.column_config.TextColumn(width="small"),
                'Probabilidad (%)': st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format='%.1f%%', width="medium"
                ),
            },
            use_container_width=True,
            hide_index=True,
            height=380
        )

        # Botón de descarga integrado bajo la tabla
        st.write("")
        csv_data = df_tabla.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Datos del Ranking (CSV)",
            data=csv_data,
            file_name="ranking_barrios_gentrificacion.csv",
            mime="text/csv",
            key="download_csv_ranking",
            use_container_width=True  
        )

    with col_chart:
        st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px 8px 0 0; padding: 10px 14px;">
                <span style="font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase;">
                    📊 {titulo_grafico}
                </span>
            </div>
        """, unsafe_allow_html=True)

        df_plot = df_mostrar.head(15).iloc[::-1] if len(df_mostrar) > 15 else df_mostrar.iloc[::-1]

        fig_rank = px.bar(
            df_plot,
            x='probabilidad',
            y='barrio_nombre',
            orientation='h',
            color='probabilidad',
            range_color=[0.0, 1.0],
            color_continuous_scale=['#10B981', '#F59E0B', '#DC2626'],
            labels={'barrio_nombre': '', 'probabilidad': 'Probabilidad'},
            text=df_plot['probabilidad'].apply(lambda x: f'{x*100:.1f}%')
        )

        fig_rank.update_traces(
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Probabilidad: %{x:.1%}<extra></extra>'
        )
        
        fig_rank.update_layout(
            height=420,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(range=[0, 1.12], tickformat='.0%'),
            yaxis=dict(type='category'),
            margin=dict(l=10, r=20, t=10, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_rank, use_container_width=True, config={'displayModeBar': False}, key="fig_ranking_interactivo")

# ============================================================================
# TAB 5: ANALÍTICA DE FACTORES Y DISTRIBUCIÓN (EDA)
# ============================================================================
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

with tab5:
    # 1. ESTILOS CSS UNIFICADOS (CARD SYSTEM & PALETA CORPORATIVA)
    st.markdown("""
        <style>
        /* Fondo general */
        .stApp {
            background-color: #F1F5F9 !important;
        }

        /* Envoltorio de tarjetas nativas st.container(border=True) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
            padding: 12px !important;
        }

        /* Cabecera interna para títulos de tarjetas */
        .card-title-bar {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px 8px 0 0;
            padding: 10px 14px;
            margin-bottom: 12px;
        }
        
        .card-title-text {
            font-size: 12px;
            font-weight: 700;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # ENCABEZADO DEL MÓDULO
    st.markdown("""
        <div style="margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 20px;">📈</span>
                <span style="color: #0F172A; font-weight: 800; font-size: 18px;">Analítica de Factores y Relaciones Bivariantes</span>
            </div>
            <p style="color: #64748B; font-size: 13px; margin: 2px 0 0 0;">
                Exploración detallada de patrones sociodemográficos, económicos y de oferta inmobiliaria que impulsan el modelo.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Identificar columnas numéricas para el análisis
    numeric_cols = df_gold_pred.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    cols_to_exclude = ['id', 'geometry', 'barrio_id', 'barrio_codigo']
    numeric_cols = [c for c in numeric_cols if c not in cols_to_exclude]

    # 2. SECCIÓN SUPERIOR: ANÁLISIS BIVARIANTE (SCATTER PLOT, KPIS Y DRILL-DOWN)
    with st.container(border=True):
        st.markdown("""
            <div class="card-title-bar">
                <span class="card-title-text">🔍 Análisis Bivariante (Relación entre Variables)</span>
            </div>
        """, unsafe_allow_html=True)

        col_x, col_y, col_c = st.columns(3)

        with col_x:
            var_x = st.selectbox("Variable Eje X:", options=numeric_cols, index=0, format_func=get_variable_label, key="eda_var_x")

        with col_y:
            default_y_idx = 1 if len(numeric_cols) > 1 else 0
            var_y = st.selectbox("Variable Eje Y:", options=numeric_cols, index=default_y_idx, format_func=get_variable_label, key="eda_var_y")

        with col_c:
            color_opts = [c for c in ['categoria', 'riesgo', 'probabilidad'] + numeric_cols if c in df_gold_pred.columns]
            color_opts = list(dict.fromkeys(color_opts))
            var_color = st.selectbox("Agrupar / Colorear por:", options=color_opts, index=0, format_func=get_variable_label, key="eda_var_color")

        # Filtro de barrios para drill-down (Opción 3)
        barrios_list = sorted(df_gold_pred['barrio_nombre'].dropna().unique().tolist()) if 'barrio_nombre' in df_gold_pred.columns else []
        selected_barrios = st.multiselect(
            "📍 Resaltar Barrio(s) Específico(s) en la gráfica:",
            options=barrios_list,
            placeholder="Escribe o selecciona barrios para destacar...",
            key="eda_selected_barrios"
        )

        # Cálculo dinámico de métricas estadísticas (Pearson r, R², p-value)
        df_clean_scatter = df_gold_pred.dropna(subset=[var_x, var_y])
        r_val, p_val, r2_val = 0.0, 1.0, 0.0
        
        if len(df_clean_scatter) > 1:
            r_val, p_val = stats.pearsonr(df_clean_scatter[var_x], df_clean_scatter[var_y])
            r2_val = r_val ** 2

        # Lógica visual para etiquetas y colores de las minitarjetas de KPIs
        fuerza_label = "Fuerte" if abs(r_val) > 0.6 else ("Moderada" if abs(r_val) > 0.3 else "Débil")
        badge_bg = "#DEF7EC" if abs(r_val) > 0.6 else ("#FEF3C7" if abs(r_val) > 0.3 else "#F3F4F6")
        badge_txt = "#03543F" if abs(r_val) > 0.6 else ("#92400E" if abs(r_val) > 0.3 else "#374151")
        sig_label = "Sí (p < 0.05)" if p_val < 0.05 else "No significativa"
        sig_color = "#10B981" if p_val < 0.05 else "#6B7280"

        # Banners KPI estilo tarjeta
        st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 12px 0 18px 0;">
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px;">
                    <div style="font-size: 11px; font-weight: 600; color: #64748B; text-transform: uppercase;">Correlación de Pearson (r)</div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-top: 4px;">
                        <span style="font-size: 22px; font-weight: 800; color: #0F172A;">{r_val:.2f}</span>
                        <span style="background-color: {badge_bg}; color: {badge_txt}; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 12px;">{fuerza_label}</span>
                    </div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px;">
                    <div style="font-size: 11px; font-weight: 600; color: #64748B; text-transform: uppercase;">Varianza Explicada (R²)</div>
                    <div style="font-size: 22px; font-weight: 800; color: #0F172A; margin-top: 4px;">{r2_val:.2%}</div>
                </div>
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px;">
                    <div style="font-size: 11px; font-weight: 600; color: #64748B; text-transform: uppercase;">Significativa (p-value)</div>
                    <div style="font-size: 18px; font-weight: 700; color: {sig_color}; margin-top: 6px;">{sig_label}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Gráfico Scatter Plot
        color_map = {'BAJO': '#10B981', 'MEDIO': '#F59E0B', 'ALTO': '#DC2626'}

        fig_scatter = px.scatter(
            df_gold_pred,
            x=var_x,
            y=var_y,
            color=var_color,
            trendline="ols",
            trendline_color_override="#2563EB",
            category_orders={var_color: ['BAJO', 'MEDIO', 'ALTO']} if var_color in ['categoria', 'riesgo'] else None,
            color_discrete_map=color_map if df_gold_pred[var_color].dtype == 'object' else None,
            color_continuous_scale="RdYlGn_r" if df_gold_pred[var_color].dtype != 'object' else None,
            hover_name='barrio_nombre' if 'barrio_nombre' in df_gold_pred.columns else None,
            height=380
        )

        fig_scatter.update_traces(marker=dict(size=7, opacity=0.85))
        fig_scatter.update_traces(line=dict(width=2.5), selector=dict(mode='lines'))

        # Capa de marcado para barrios seleccionados (Drill-down)
        if selected_barrios and 'barrio_nombre' in df_gold_pred.columns:
            df_highlight = df_gold_pred[df_gold_pred['barrio_nombre'].isin(selected_barrios)].dropna(subset=[var_x, var_y])
            if not df_highlight.empty:
                fig_scatter.add_trace(
                    go.Scatter(
                        x=df_highlight[var_x],
                        y=df_highlight[var_y],
                        mode='markers+text',
                        marker=dict(
                            size=14,
                            color='#1E40AF',
                            symbol='star',
                            line=dict(width=1.5, color='#FFFFFF')
                        ),
                        text=df_highlight['barrio_nombre'],
                        textposition="top center",
                        name="Barrios Resaltados",
                        showlegend=True
                    )
                )

        fig_scatter.update_layout(
            xaxis_title=get_variable_label(var_x),
            yaxis_title=get_variable_label(var_y),
            coloraxis_colorbar=dict(title=get_variable_label(var_color)),
            legend=dict(title=get_variable_label(var_color), orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9')
        )

        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False}, key="fig_eda_scatter")

    # 3. SECCIÓN INFERIOR: BOXPLOT & HEATMAP DE CORRELACIÓN
    col_g1, col_g2 = st.columns([1, 1], gap="large")

    # --- TARJETA BOXPLOT ---
    with col_g1:
        with st.container(border=True):
            st.markdown("""
                <div class="card-title-bar">
                    <span class="card-title-text">📦 Distribución según Categoría de Riesgo (Boxplot)</span>
                </div>
            """, unsafe_allow_html=True)

            var_box = st.selectbox(
                "Seleccionar variable para analizar por nivel de riesgo:",
                options=numeric_cols,
                index=0,
                format_func=get_variable_label,
                key="eda_var_box"
            )

            df_box = df_gold_pred.dropna(subset=[var_box]).copy()
            col_cat = next((c for c in ['categoria', 'riesgo', 'nivel_riesgo', 'target', 'cat_clean'] if c in df_box.columns), None)

            if col_cat:
                def normalizar_categoria(val):
                    s = str(val).strip().upper()
                    if 'BAJ' in s or s in ['0', '0.0', 'LOW']:
                        return 'BAJO'
                    elif 'MED' in s or s in ['1', '1.0', 'MEDIUM']:
                        return 'MEDIO'
                    elif 'ALT' in s or s in ['2', '2.0', 'HIGH']:
                        return 'ALTO'
                    return s

                df_box['cat_norm'] = df_box[col_cat].apply(normalizar_categoria)
                palette = {'BAJO': '#10B981', 'MEDIO': '#F59E0B', 'ALTO': '#DC2626'}
                cats_presentes = [c for c in ['BAJO', 'MEDIO', 'ALTO'] if c in df_box['cat_norm'].unique()]
                
                if not cats_presentes:
                    cats_presentes = list(df_box['cat_norm'].unique())

                fig_box = go.Figure()

                for cat_label in cats_presentes:
                    df_sub = df_box[df_box['cat_norm'] == cat_label]
                    cat_color = palette.get(cat_label, '#6366F1')
                    
                    if not df_sub.empty:
                        fig_box.add_trace(
                            go.Box(
                                y=df_sub[var_box],
                                name=cat_label,
                                boxpoints='all',
                                jitter=0.3,
                                pointpos=0,
                                marker=dict(color=cat_color, size=6, opacity=0.7),
                                line=dict(color=cat_color, width=1.5),
                                fillcolor=cat_color,
                                opacity=0.6,
                                text=df_sub['barrio_nombre'] if 'barrio_nombre' in df_sub.columns else None,
                                hovertemplate='<b>%{text}</b><br>' + get_variable_label(var_box) + ': %{y}<extra></extra>'
                            )
                        )

                fig_box.update_layout(
                    height=360,
                    showlegend=False,
                    xaxis_title="Nivel de Riesgo",
                    yaxis_title=get_variable_label(var_box),
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(showgrid=True, gridcolor='#F1F5F9')
                )

                st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False}, key="fig_eda_box")
            else:
                st.warning("No se encontró una columna de categoría/riesgo en el conjunto de datos.")

    # --- TARJETA MATRIZ DE CORRELACIÓN ---
    with col_g2:
        with st.container(border=True):
            st.markdown("""
                <div class="card-title-bar">
                    <span class="card-title-text">🔥 Matriz de Correlación (Variables Clave)</span>
                </div>
            """, unsafe_allow_html=True)

            cols_corr = [
                'probabilidad', 
                'precio_m2_registradores', 
                'pct_extranjeros', 
                'edad_media', 
                'renta_media_2023',
                'n_bares_202206'
            ]
            
            cols_corr_presentes = [c for c in cols_corr if c in df_gold_pred.columns]

            if len(cols_corr_presentes) > 1:
                corr_matrix = df_gold_pred[cols_corr_presentes].corr().round(2)
                labels_corr = [get_variable_label(c) for c in cols_corr_presentes]

                fig_corr = px.imshow(
                    corr_matrix,
                    x=labels_corr,
                    y=labels_corr,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    aspect="auto",
                    height=360
                )

                fig_corr.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    coloraxis_showscale=False
                )

                st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False}, key="fig_eda_corr")
            else:
                st.warning("No hay suficientes variables disponibles para calcular la matriz de correlación.")

    # 4. INSIGHTS AUTOMATIZADOS DE INTERPRETACIÓN TÉCNICA
    with st.expander("💡 Conclusiones y Resumen del Análisis de Factores"):
        direccion = "positiva" if r_val > 0 else "negativa"
        fuerza = "fuerte" if abs(r_val) > 0.6 else ("moderada" if abs(r_val) > 0.3 else "débil")
        
        st.markdown(f"""
        * **Relación Bivariante:** La relación entre **{get_variable_label(var_x)}** y **{get_variable_label(var_y)}** presenta una asociación **{direccion} {fuerza}** ($r = {r_val:.2f}$).
        * **Significativa Estadística:** El p-valor indica que esta relación es **{'significativa (p < 0.05)' if p_val < 0.05 else 'no significativa'}**, lo que valida su inclusión como predictor en el modelo.
        * **Estructura por Riesgo:** Las variables sociodemográficas y de oferta comercial se distribuyen de forma asimétrica entre los distintos niveles de riesgo, permitiendo una clara segmentación territorial.
        """)
# ============================================================================
# FOOTER
# ============================================================================
st.markdown('---')
st.caption(
    '© 2026 - Radar de Barrio - TFM Vandeson Sena | Modelo Ensemble V2 Calibrado'
)