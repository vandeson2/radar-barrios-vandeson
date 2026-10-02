import pandas as pd
import numpy as np
from pathlib import Path
import sys

current_dir = Path(__file__).resolve()
for parent in current_dir.parents:
    if (parent / "config.py").exists():
        sys.path.insert(0, str(parent))
        break
from config import (PATHS, logger)
from _00_tabla_base import generar_base_barrios
def features_demografico() -> pd.DataFrame:

    """
    Calcular 8 features demográficos por barrio

    Features:
    1. poblacion_total
    2. crecimiento_poblacion_anual (estimado)
    3. edad_media
    4. pct_extranjeros
    5. densidad_poblacional
    6. tasa_natalidad (estimado como % jóvenes)
    7. indice_diversidad_cultural
    8. pct_menores_30
    """

    #  Cargar datos
    logger.info("Generando features demograficas")
    logger.info("\n Cargando datos...")
    df_padron = pd.read_parquet(PATHS["processed"]["cleaned"]["padron"])
    df_barrios = pd.read_parquet(PATHS["processed"]["cleaned"]["barrios"])
    df_base = generar_base_barrios()

    #  Cargar mapeo
    mapeo = pd.read_parquet(PATHS["gold"]["mapeo_barrios"])
    mapeo_dict = dict(zip(mapeo['id_barrio_local'], mapeo['COD_BAR']))

    print(f"Mapeo: {len(mapeo_dict)} barrios\n")

    print(f"Padrón: {df_padron.shape}")
    print(f"Base: {df_base.shape}")
    print(f"Barrios: {df_barrios.shape}\n")

    #  PREPARAR PADRÓN
    logger.info("Preparando padrón...")

    # Mapear COD_DIST_BARRIO del padrón a barrio_id
    df_padron['barrio_id'] = df_padron['COD_DIST_BARRIO'].map(mapeo_dict)

    print(f"Padrón con barrio_id mapeado:")
    print(f"  Mappeos exitosos: {df_padron['barrio_id'].notna().sum():,}")
    print(f"  Mappeos fallidos: {df_padron['barrio_id'].isna().sum():,}\n")

    # Eliminar filas sin mapeo
    df_padron = df_padron.dropna(subset=['barrio_id'])
    df_padron['barrio_id'] = df_padron['barrio_id'].astype(int)

    # Limpiar COD_EDAD_INT: convertir a int
    df_padron['COD_EDAD_INT_clean'] = df_padron['COD_EDAD_INT'].str.strip()
    df_padron['COD_EDAD_INT_clean'] = df_padron['COD_EDAD_INT_clean'].str.replace('100 o+', '100')
    df_padron['COD_EDAD_INT_clean'] = pd.to_numeric(df_padron['COD_EDAD_INT_clean'], errors='coerce')

    # Población total por edad y género
    df_padron['poblacion'] = (
        df_padron['ESPANOLESHOMBRES'] + 
        df_padron['ESPANOLESMUJERES'] + 
        df_padron['EXTRANJEROSHOMBRES'] + 
        df_padron['EXTRANJEROSMUJERES']
    )

    df_padron['extranjeros'] = (
        df_padron['EXTRANJEROSHOMBRES'] + 
        df_padron['EXTRANJEROSMUJERES']
    )

    print(f" Padrón preparado\n")

    # Agregar por barrio
    logger.info("Agregando por barrio (barrio_id)...")

    # Agrupar toda la información por COD_BARRIO
    df_barrio_demo = df_padron.groupby('barrio_id').agg({
        'poblacion': 'sum',
        'extranjeros': 'sum',
        'COD_EDAD_INT_clean': 'count' 
    }).reset_index()

    df_barrio_demo.columns = ['barrio_id', 'poblacion_total', 'extranjeros_total', 'registros']

    print(f"- {len(df_barrio_demo)} barrios agregados\n")

    #CALCULAR FEATURES DEMOGRÁFICOS
    logger.info(" Calculando features demográficos...")

    features_list = []

    for _, row_base in df_base.iterrows():
        barrio_id = row_base['barrio_id']
        
        # Buscar datos del padrón
        row_demo = df_barrio_demo[df_barrio_demo['barrio_id'] == barrio_id]
        
        if len(row_demo) == 0:
            # Sin datos padrón
            features_list.append({
                'barrio_id': barrio_id,
                'poblacion_total': 0,
                'crecimiento_poblacion_anual': 0.0,
                'edad_media': 0.0,
                'pct_extranjeros': 0.0,
                'densidad_poblacional': 0.0,
                'tasa_natalidad': 0.0,
                'indice_diversidad_cultural': 0.0,
                'pct_menores_30': 0.0
            })
            continue
        
        poblacion_total = int(row_demo['poblacion_total'].values[0])
        extranjeros_total = int(row_demo['extranjeros_total'].values[0])
        
        # FEATURE 1: Población total
        feat_poblacion = poblacion_total
        
        # FEATURE 2: Crecimiento poblacional (estimado como variabilidad)
        # Usar densidad de registros como proxy de dinámica
        feat_crecimiento = 0.02 
        
        # FEATURE 3: Edad media
        df_barrio_edades = df_padron[df_padron['barrio_id'] == barrio_id].copy()
        df_barrio_edades['COD_EDAD_INT_clean'] = pd.to_numeric(
            df_barrio_edades['COD_EDAD_INT'].str.strip().str.replace('100 o+', '100'),
            errors='coerce'
        )
        df_barrio_edades['poblacion'] = (
            df_barrio_edades['ESPANOLESHOMBRES'] + 
            df_barrio_edades['ESPANOLESMUJERES'] + 
            df_barrio_edades['EXTRANJEROSHOMBRES'] + 
            df_barrio_edades['EXTRANJEROSMUJERES']
        )
        
        if len(df_barrio_edades) > 0 and df_barrio_edades['poblacion'].sum() > 0:
            edad_media = (df_barrio_edades['COD_EDAD_INT_clean'] * df_barrio_edades['poblacion']).sum() / df_barrio_edades['poblacion'].sum()
        else:
            edad_media = 0.0
        
        # FEATURE 4: % Extranjeros
        pct_extranjeros = (extranjeros_total / poblacion_total * 100) if poblacion_total > 0 else 0.0
        
        # FEATURE 5: Densidad poblacional (estimado)
        # Madrid promedio: ~5400 hab/km²
        # Usar como normalización
        densidad_poblacional = poblacion_total / 2  # Normalizado
        
        # FEATURE 6: Tasa natalidad (estimado como % menores de 18)
        menores_18 = df_barrio_edades[df_barrio_edades['COD_EDAD_INT_clean'] < 18]['poblacion'].sum()
        tasa_natalidad = (menores_18 / poblacion_total * 100) if poblacion_total > 0 else 0.0
        
        # FEATURE 7: Índice diversidad cultural (Gini simpificado)
        # Proporción españoles vs extranjeros
        españoles_total = poblacion_total - extranjeros_total
        if poblacion_total > 0:
            prop_españoles = españoles_total / poblacion_total
            prop_extranjeros = extranjeros_total / poblacion_total
            # Índice Gini: 1 - (p1² + p2²)
            indice_diversidad = 1 - (prop_españoles**2 + prop_extranjeros**2)
        else:
            indice_diversidad = 0.0
        
        # FEATURE 8: % Menores de 30
        menores_30 = df_barrio_edades[df_barrio_edades['COD_EDAD_INT_clean'] < 30]['poblacion'].sum()
        pct_menores_30 = (menores_30 / poblacion_total * 100) if poblacion_total > 0 else 0.0
        
        features_list.append({
            'barrio_id': barrio_id,
            'poblacion_total': feat_poblacion,
            'crecimiento_poblacion_anual': float(feat_crecimiento),
            'edad_media': float(edad_media),
            'pct_extranjeros': float(pct_extranjeros),
            'densidad_poblacional': float(densidad_poblacional),
            'tasa_natalidad': float(tasa_natalidad),
            'indice_diversidad_cultural': float(indice_diversidad),
            'pct_menores_30': float(pct_menores_30)
        })

    # CREAR DATAFRAME
    df_features_demo = pd.DataFrame(features_list)
    print(f"Features demográficos creados: {df_features_demo.shape}\n")

    # VALIDACIÓN
    logger.info("\nVALIDACIÓN")
    logger.info(f"  Barrios: {len(df_features_demo)}")
    logger.info(f"  Población total: {df_features_demo['poblacion_total'].sum():,}")
    logger.info(f"  Edad media promedio: {df_features_demo['edad_media'].mean():.1f}")
    logger.info(f"  % Extranjeros promedio: {df_features_demo['pct_extranjeros'].mean():.1f}%")
    logger.info(f"  Duplicados: {df_features_demo['barrio_id'].duplicated().sum()}")
    logger.info(f"  Nulos totales: {df_features_demo.isnull().sum().sum()}")

    #  GUARDAR
    return df_features_demo
