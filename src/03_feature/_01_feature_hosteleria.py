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

def  features_hosteleria () -> pd.DataFrame: 
    #  Cargar datos
    logger.info("\nCargando datos...")
    df_base = generar_base_barrios()
    df_consolidado = pd.read_parquet(PATHS["processed"]["cleaned"]["hosteleria_consolidado"])
    mapeo = pd.read_parquet(PATHS["gold"]["mapeo_barrios"])

    print(f"Consolidado: {df_consolidado.shape}")
    print(f"Mapeo: {mapeo.shape}")
    print(f"Base barrios: {df_base.shape}\n")

    #Filtrar hostelería
    logger.info("Filtrando solo hostelería...")
    df_host = df_consolidado[df_consolidado['es_hosteleria'] == True].copy()
    print(f"Registros hostelería: {len(df_host):,}\n")

    # CREAR MAPEO: id_barrio_local → COD_BAR
    logger.info("Creando mapeo consolidado → base...")
    mapeo_dict = dict(zip(mapeo['id_barrio_local'], mapeo['COD_BAR']))
    df_host['cod_bar_mapped'] = df_host['id_barrio_local'].map(mapeo_dict)

    # Verificar mapeo
    print(f"Registros con mapeo válido: {df_host['cod_bar_mapped'].notna().sum():,}")
    print(f"Registros sin mapeo: {df_host['cod_bar_mapped'].isna().sum():,}\n")

    # USAR BARRIOS MAPEADOS
    df_host_mapped = df_host[df_host['cod_bar_mapped'].notna()].copy()
    df_host_mapped['barrio_id'] = df_host_mapped['cod_bar_mapped'].astype(int)

    #CALCULAR FEATURES POR BARRIO
    logger.info("Calculando features por barrio...")

    features_list = []

    for barrio_id in df_base['barrio_id'].values:
        df_barrio = df_host_mapped[df_host_mapped['barrio_id'] == barrio_id].copy()
        
        if len(df_barrio) == 0:
            # Sin datos hostelería
            features_list.append({
                'barrio_id': barrio_id,
                'n_bares_202206': 0,
                'n_bares_202606': 0,
                'velocidad_crecimiento_anual': 0.0,
                'aceleracion_crecimiento': 0.0,
                'media_movil_12m': 0.0,
                'tendencia_54m': 0.0,
                'variabilidad_bares': 0.0,
                'cambio_acumulado_pct': 0.0,
                'diversidad_categorias': 0,
                'bares_por_1000hab': 0.0
            })
            continue
        
        # Agrupar por mes
        df_barrio['year_month'] = df_barrio['fecha'].dt.to_period('M')
        bares_por_mes = df_barrio.groupby('year_month').size()
        
        # FEATURES
        n_bares_202206 = len(df_barrio[df_barrio['fecha'].dt.to_period('M') == '2022-02'])
        n_bares_202606 = len(df_barrio[df_barrio['fecha'].dt.to_period('M') == '2026-06'])
        
        if n_bares_202206 > 0:
            velocidad_anual = (n_bares_202606 - n_bares_202206) / n_bares_202206 / (54/12)
        else:
            velocidad_anual = 0.0
        
        if len(bares_por_mes) > 24:
            vel_ultimos_12 = (bares_por_mes.iloc[-1] - bares_por_mes.iloc[-12]) / bares_por_mes.iloc[-12] if bares_por_mes.iloc[-12] > 0 else 0
            vel_primeros_12 = (bares_por_mes.iloc[11] - bares_por_mes.iloc[0]) / bares_por_mes.iloc[0] if bares_por_mes.iloc[0] > 0 else 0
            aceleracion = (vel_ultimos_12 - vel_primeros_12) / abs(vel_primeros_12) if vel_primeros_12 != 0 else 0.0
        else:
            aceleracion = 0.0
        
        media_movil_12 = float(bares_por_mes.iloc[-12:].mean()) if len(bares_por_mes) >= 12 else float(bares_por_mes.mean())
        
        if len(bares_por_mes) > 1:
            x = np.arange(len(bares_por_mes))
            y = bares_por_mes.values
            tendencia = float(np.polyfit(x, y, 1)[0])
        else:
            tendencia = 0.0
        
        cambios_mensuales = bares_por_mes.diff().dropna()
        variabilidad = float(cambios_mensuales.std()) if len(cambios_mensuales) > 0 else 0.0
        
        cambio_acum = (n_bares_202606 - n_bares_202206) / n_bares_202206 * 100 if n_bares_202206 > 0 else 0.0
        
        diversidad = int(df_barrio['desc_division'].nunique())
        
        features_list.append({
            'barrio_id': barrio_id,
            'n_bares_202206': int(n_bares_202206),
            'n_bares_202606': int(n_bares_202606),
            'velocidad_crecimiento_anual': float(velocidad_anual),
            'aceleracion_crecimiento': float(aceleracion),
            'media_movil_12m': float(media_movil_12),
            'tendencia_54m': float(tendencia),
            'variabilidad_bares': float(variabilidad),
            'cambio_acumulado_pct': float(cambio_acum),
            'diversidad_categorias': diversidad,
            'bares_por_1000hab': 0.0 
        })

    # CREAR DATAFRAME
    df_features_host = pd.DataFrame(features_list)
    print(f"Features hostelería creadas: {df_features_host.shape}\n")

    #VALIDACIÓN
    logger.info("\nVALIDACIÓN")
    logger.info(f"  Barrios: {len(df_features_host)}")
    logger.info(f"  Barrios con >0 bares: {(df_features_host['n_bares_202606'] > 0).sum()}")
    logger.info(f"  Barrios sin bares: {(df_features_host['n_bares_202606'] == 0).sum()}")
    logger.info(f"  Duplicados: {df_features_host['barrio_id'].duplicated().sum()}")
    logger.info(f"  Nulos totales: {df_features_host.isnull().sum().sum()}")

    return df_features_host




