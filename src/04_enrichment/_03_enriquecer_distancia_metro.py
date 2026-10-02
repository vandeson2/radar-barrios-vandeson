from pathlib import Path
import numpy as np
import pandas as pd

# Diccionario de estaciones
ESTACIONES_METRO = {
    "Sol": (40.4169, -3.7038),
    "Plaza Mayor": (40.4150, -3.7072),
    "Callao": (40.4200, -3.7052),
    "Gran Vía": (40.4207, -3.7049),
    "Banco de España": (40.4176, -3.6930),
    "Atocha": (40.4084, -3.6931),
    "Moncloa": (40.4489, -3.7364),
    "Ventas": (40.4417, -3.6000),
    "Goya": (40.4267, -3.6564),
    "Retiro": (40.4114, -3.6810),
    "Colón": (40.4255, -3.6809),
    "Ópera": (40.4193, -3.7151),
    "La Latina": (40.4087, -3.7192),
    "Tirso de Molina": (40.4121, -3.7093),
    "Legazpi": (40.3939, -3.7150),
    "Arganzuela": (40.3980, -3.7250),
    "Pirámides": (40.3930, -3.7122),
    "Puerta de Atocha": (40.4063, -3.6918),
    "Palos de la Frontera": (40.3806, -3.6944),
    "Pacífico": (40.4167, -3.6667),
    "Ibiza": (40.4210, -3.6744),
    "Sainz de Baranda": (40.4113, -3.6673),
    "O'Donnell": (40.4178, -3.6752),
    "Diego de León": (40.4272, -3.6710),
    "Núñez de Balboa": (40.4330, -3.6732),
    "Serrano": (40.4370, -3.6757),
    "Príncipe de Vergara": (40.4398, -3.6705),
    "Rubén Darío": (40.4457, -3.6641),
    "Castellana": (40.4442, -3.6333),
    "Cuatro Caminos": (40.4736, -3.7131),
    "Bilbao": (40.4487, -3.6963),
    "Tribunal": (40.4441, -3.7036),
    "Alonso Martínez": (40.4418, -3.6937),
    "Chueca": (40.4268, -3.7016),
    "Santo Domingo": (40.4278, -3.7148),
    "Noviciado": (40.4378, -3.7188),
    "Argüelles": (40.4411, -3.7322),
    "Ciudad Universitaria": (40.4489, -3.7333),
    "Avenida de América": (40.4554, -3.6555),
    "Chamartin": (40.4616, -3.6346),
    "Santiago Bernabéu": (40.4531, -3.6877),
    "Plaza de Castilla": (40.4648, -3.6892),
}


def _calcular_haversine_matriz(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calcula la matriz de distancias Haversine (en km) entre barrios y estaciones."""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad[None, :] - lat1_rad[:, None]
    dlon = lon2_rad[None, :] - lon1_rad[:, None]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad[:, None])
        * np.cos(lat2_rad[None, :])
        * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def calcular_accesibilidad_metro(
    df: pd.DataFrame,
    col_barrio: str = "barrio_nombre",
    col_lat: str = "latitud",
    col_lon: str = "longitud",
) -> pd.DataFrame:
    """Función Pura: Recibe un DataFrame con coordenadas y calcula las distancias a metro."""
    if col_lat not in df.columns or col_lon not in df.columns:
        raise KeyError(
            f"Faltan columnas de coordenadas '{col_lat}'/'{col_lon}' en el DataFrame."
        )

    lats_barrios = df[col_lat].to_numpy()
    lons_barrios = df[col_lon].to_numpy()

    coords_metro = list(ESTACIONES_METRO.values())
    lats_metro, lons_metro = map(np.array, zip(*coords_metro))

    matriz_distancias = _calcular_haversine_matriz(
        lats_barrios, lons_barrios, lats_metro, lons_metro
    )

    dist_min = np.min(matriz_distancias, axis=1)

    return pd.DataFrame(
        {
            "barrio_nombre": (
                df[col_barrio] if col_barrio in df.columns else df.index
            ),
            "distancia_metro_km": np.round(dist_min, 2),
            "tiempo_pie_a_metro_min": (dist_min * 1000 / 200).astype(int),
            "accesibilidad_metro": np.select(
                condlist=[dist_min < 1.0, dist_min < 2.0],
                choicelist=[1, 2],
                default=3,
            ),
        }
    )


def procesar_csv_o_parquet(
    path_entrada: str = "data/barrios_madrid.csv",
    
) -> pd.DataFrame:
    """Función Orquestadora: Carga la ruta parametrizada (CSV o Parquet),

    aplica la transformación y guarda los datos.
    """
    path_obj = Path(path_entrada)

    if not path_obj.exists():
        raise FileNotFoundError(
            f" No se encontró el archivo en: {path_entrada}"
        )

    # Cargar según extensión
    if path_obj.suffix == ".parquet":
        df_origen = pd.read_parquet(path_entrada)
    else:
        df_origen = pd.read_csv(path_entrada)

    print(f" Cargado '{path_entrada}' con {len(df_origen)} registros.")

    # Calcular
    df_resultado = calcular_accesibilidad_metro(df_origen)

    # Guardar resultado
    """
    Path(path_salida).parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(path_salida, index=False)
    print(f"Guardado resultado en '{path_salida}'")
    """
    return df_resultado