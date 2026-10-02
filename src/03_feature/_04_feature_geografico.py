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
def features_geografica() -> pd.DataFrame:
    """
     Calcular 3 features geográficos por barrio

    Features:
    1. distancia_centro_km (distancia desde Puerta del Sol)
    2. proximidad_estaciones_metro (estimado por densidad)
    3. zona (Norte/Sur/Este/Oeste)
    """
    #  Cargar datos
    logger.info("Generando features geografica...")
    logger.info("\nCargando datos...")
    df_cons = pd.read_parquet(PATHS["processed"]["cleaned"]["hosteleria_consolidado"])
    df_base = generar_base_barrios()
    mapeo = pd.read_parquet(PATHS["gold"]["mapeo_barrios"])

    print(f"Consolidado: {df_cons.shape}\n")

    # 2. PREPARAR CONSOLIDADO
    logger.info("Preparando consolidado...")

    mapeo_dict = dict(zip(mapeo['id_barrio_local'], mapeo['COD_BAR']))
    df_cons['barrio_id'] = df_cons['id_barrio_local'].map(mapeo_dict)
    df_cons = df_cons.dropna(subset=['barrio_id'])
    df_cons['barrio_id'] = df_cons['barrio_id'].astype(int)

    print(f"Consolidado mapeado: {len(df_cons):,} registros\n")

    # 3. CALCULAR CENTROIDES
    logger.info("Calculando centroides por barrio...")

    CENTRO_X = 440000
    CENTRO_Y = 4474000

    centroides = df_cons.groupby('barrio_id').agg({
        'coordenada_x_local': 'mean',
        'coordenada_y_local': 'mean',
        'local_id': 'count'
    }).reset_index()

    centroides.columns = ['barrio_id', 'coord_x', 'coord_y', 'n_locales']

    # Calcular distancia Euclidiana en UTM
    centroides['distancia_utm'] = np.sqrt(
        (centroides['coord_x'] - CENTRO_X)**2 + 
        (centroides['coord_y'] - CENTRO_Y)**2
    )

    # Normalizar distancias a rango 0-1
    distancia_min = centroides['distancia_utm'].min()
    distancia_max = centroides['distancia_utm'].max()

    centroides['distancia_normalizada'] = (
        (centroides['distancia_utm'] - distancia_min) / 
        (distancia_max - distancia_min)
    )

    print(f"Centroides calculados: {len(centroides)}")
    print(f"Distancia min (UTM): {distancia_min:.0f}")
    print(f"Distancia max (UTM): {distancia_max:.0f}\n")

    # 4. CALCULAR FEATURES GEOGRÁFICOS
    logger.info("Calculando features geográficos...")

    features_list = []

    for _, row_base in df_base.iterrows():
        barrio_id = row_base['barrio_id']
        
        centroide = centroides[centroides['barrio_id'] == barrio_id]
        
        if len(centroide) == 0:
            features_list.append({
                'barrio_id': barrio_id,
                'distancia_centro': 0.0,
                'proximidad_estaciones_metro': 0.0,
                'zona': 'Unknown'
            })
            continue
        
        coord_x = float(centroide['coord_x'].values[0])
        coord_y = float(centroide['coord_y'].values[0])
        distancia_norm = float(centroide['distancia_normalizada'].values[0])
        n_locales = int(centroide['n_locales'].values[0])
        
        # FEATURE 1: Distancia al centro (0-1, donde 0=centro, 1=periferia)
        # Invertir: 1 - distancia para que 0=periferia, 1=centro
        distancia_centro = 1 - distancia_norm
        
        # FEATURE 2: Proximidad a estaciones metro
        # Usar densidad de locales como proxy
        densidad = n_locales / 100.0
        proximidad_metro = min(densidad / (densidad + 1), 1.0)
        
        # FEATURE 3: Zona
        zona_ns = 'Norte' if coord_y > CENTRO_Y else 'Sur'
        zona_ew = 'Este' if coord_x > CENTRO_X else 'Oeste'
        zona = f'{zona_ns}-{zona_ew}'
        
        features_list.append({
            'barrio_id': barrio_id,
            'distancia_centro': float(distancia_centro),
            'proximidad_estaciones_metro': float(proximidad_metro),
            'zona': zona
        })

    # 5. CREAR DATAFRAME
    df_features_geo = pd.DataFrame(features_list)
    print(f"Features geográficos creados: {df_features_geo.shape}\n")
    # 7. VALIDACIÓN
    logger.info("\nVALIDACIÓN")
    logger.info(f"  Barrios: {len(df_features_geo)}")
    logger.info(f"  Distancia_centro min: {df_features_geo['distancia_centro'].min():.3f} (periferia)")
    logger.info(f"  Distancia_centro max: {df_features_geo['distancia_centro'].max():.3f} (centro)")
    logger.info(f"  Distancia_centro promedio: {df_features_geo['distancia_centro'].mean():.3f}")
    logger.info(f"  Proximidad_metro min: {df_features_geo['proximidad_estaciones_metro'].min():.3f}")
    logger.info(f"  Proximidad_metro max: {df_features_geo['proximidad_estaciones_metro'].max():.3f}")
    logger.info(f"  Zonas únicas: {df_features_geo['zona'].nunique()}")
    logger.info(f"  Duplicados: {df_features_geo['barrio_id'].duplicated().sum()}")
    logger.info(f"  Nulos: {df_features_geo.isnull().sum().sum()}")

    return df_features_geo

    