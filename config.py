import logging
import sys
from pathlib import Path
from typing import Dict, List

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "data" if (PROJECT_ROOT.parent / "data").exists() else PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_OTROS = DATA_RAW / "otros"
DATA_PROCESSED = DATA_DIR if (DATA_DIR / "02_cleaned").exists() else DATA_DIR / "processed"
DATA_GOLD = DATA_DIR / "gold"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# ESTRUCTURA DE AÑOS Y MESES
# =============================================================================

ANIOS_ESTRUCTURA = {
    2022: {
        "meses": list(range(2, 13)),
        "directorio": DATA_RAW / "2022",
    },
    2023: {
        "meses": list(range(1, 13)),
        "directorio": DATA_RAW / "2023",
    },
    2024: {
        "meses": list(range(1, 13)),
        "directorio": DATA_RAW / "2024",
    },
    2025: {
        "meses": list(range(1, 13)),
        "directorio": DATA_RAW / "2025",
    },
     2026: {
        "meses": list(range(1, 7)),
        "directorio": DATA_RAW / "2026",
    },
}

ANIOS_PROCESAR = list(ANIOS_ESTRUCTURA.keys())
# =============================================================================
# ESTRUCTURA OTROS DATASSETS
# =============================================================================
DATOS_COMPLEMENTARIOS = {
    "padron": {
        "path": DATA_OTROS / "padron-municipal.csv",
        "delimiter": ";",
        "encoding": "utf-8",
        "clean_cols": True,  
    },
    "renta": {
        "path": DATA_OTROS / "Indicadores-renta-media-mediana.csv",
        "delimiter": ";",
        "encoding": "utf-8",
        "filter_madrid": True, 
    },
    "precios": {
        "path": DATA_OTROS / "precios_historico_madrid_distritos.csv",
        "delimiter": ",",
        "encoding": "utf-8",
    },
    "barrios": {
        "path": DATA_OTROS / "Barrios.csv",
        "delimiter": ";",
        "encoding": "utf-8",
        "clean_area": True,  
    }
}
# =============================================================================
# MESES
# =============================================================================

NOMBRES_MESES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

# =============================================================================
# CONFIGURACIÓN DE CSV
# =============================================================================
CSV_DELIMITER = ";"
CSV_ENCODING = "utf-8-sig"
CSV_LOW_MEMORY = False
CSV_PATTERN = "{anio}{mes:02d}_{mes_nombre}_{tipo}.csv"

#Opciones de lectura de CSV (tolerancia de errores)
CSV_READ_OPTIONS = {
    "delimiter": CSV_DELIMITER,
    "encoding": CSV_ENCODING,
    #"low_memory": CSV_LOW_MEMORY,
    "on_bad_lines": "skip",
    "engine": "python",
    
}
# =============================================================================
# ARCHIVOS PROCESADOS
# =============================================================================

PROCESSED_PATHS ={
    "consolidated": DATA_PROCESSED / "01_consolidated" / "dataset_complete.parquet",
    "consolidated_2022": DATA_PROCESSED / "01_consolidated" / "dataset_2022.parquet",
    "consolidated_2023": DATA_PROCESSED / "01_consolidated" / "dataset_2023.parquet",
    "consolidated_2024": DATA_PROCESSED / "01_consolidated" / "dataset_2024.parquet",
    "consolidated_2025": DATA_PROCESSED / "01_consolidated" / "dataset_2025.parquet",
    "consolidated_2026": DATA_PROCESSED / "01_consolidated" / "dataset_2026.parquet",
    "cleaned_2022": DATA_PROCESSED / "02_cleaned" / "dataset_cleaned_2022.parquet",
    "cleaned_2023": DATA_PROCESSED / "02_cleaned" / "dataset_cleaned_2023.parquet",
    "cleaned_2024": DATA_PROCESSED / "02_cleaned" / "dataset_cleaned_2024.parquet",
    "cleaned_2025": DATA_PROCESSED / "02_cleaned" / "dataset_cleaned_2025.parquet",
    "cleaned_2026": DATA_PROCESSED / "02_cleaned" / "dataset_cleaned_2026.parquet",
    "consolidated_padron": DATA_PROCESSED / "01_consolidated" / "dataset_padron.parquet",
    "consolidated_renta": DATA_PROCESSED / "01_consolidated" / "dataset_renta.parquet",
    "consolidated_precios": DATA_PROCESSED / "01_consolidated" / "dataset_precios.parquet",
    "consolidated_barrios": DATA_PROCESSED / "01_consolidated" / "dataset_barrios.parquet",
    "cleaned_padron": DATA_PROCESSED / "02_cleaned"/ "dataset_cleaned_padron.parquet",
    "cleaned_renta": DATA_PROCESSED / "02_cleaned"/ "dataset_cleaned_renta.parquet",
    "cleaned_precios": DATA_PROCESSED / "02_cleaned"/ "dataset_cleaned_precios.parquet",
    "cleaned_barrios": DATA_PROCESSED / "02_cleaned"/ "dataset_cleaned_barrios.parquet",
    "cleaned_all_years": DATA_PROCESSED / "02_cleaned" / "dataset_consolidado_all_years.parquet",
}   
PATHS = {
    "data":{
        "gold_test": DATA_DIR / "gold_test.parquet",
        "barrios_Madrid": DATA_DIR / "barrios_madrid_130.csv",
        "enriquecer_precios": DATA_DIR / "datos_precios_registradores_barrios.csv",
        
    },
    "processed": {
        "cleaned": {
            "hosteleria_consolidado": DATA_PROCESSED / "02_cleaned" / "dataset_consolidado_all_years.parquet",
            "barrios": DATA_PROCESSED / "02_cleaned" / "dataset_cleaned_barrios.parquet",
            "padron": DATA_PROCESSED / "02_cleaned" / "dataset_cleaned_padron.parquet",
            "renta": DATA_PROCESSED / "02_cleaned" / "dataset_cleaned_renta.parquet",
            "precios": DATA_PROCESSED/ "02_cleaned" / "dataset_cleaned_precios.parquet",
        },
    },
    "enrichment": {

    },
    "gold": {
        "base": DATA_GOLD / "01_base_barrios.parquet",
        "hosteleria": DATA_GOLD / "02_feature_ hosteleria.parquet",
        "demografico": DATA_GOLD / "03_feature_demografico.parquet",
        "economico": DATA_GOLD / "04_feature_economico.parquet",
        "geografico": DATA_GOLD / "05_feature_geografico.parquet",
        "target": DATA_GOLD / "06_target.parquet",
        "gold_final": DATA_GOLD / "gold_barrios_enriquecido.parquet",
        "gold_1": DATA_GOLD / "gold_1.parquet",
        "mapeo_barrios": DATA_GOLD / "mapeo_barrios_final.parquet"
    },
    

}
CENSO_LOCALES_COLUMNS = [
    "local_id",
    "fecha",
    "barrio",
    "barrio_codigo",
    "distrito",
    "postal",
    "longitud",
    "latitud",
    "tipo_local",
    "estado_local",
]

CENSO_ACTIVIDADES_COLUMNS = [
    "local_id",
    "fecha",
    "actividad_id",
    "descripcion_actividad",
    "epigrafe",
    "clase_actividad",
    "sector",
    "estado_actividad",
]

# =============================================================================
# IDENTIFICACIÓN DE HOSTELERÍA
# =============================================================================

PALABRAS_HOSTELERIA = [
    "BAR",
    "CAFÉ",
    "CAFETERIA",
    "RESTAURANTE",
    "COMIDA RAPIDA",
    "PIZZERIA",
    "TABERNA",
    "PUB",
    "CERVECERIA",
    "BODEGA",
    "TERRAZA",
    "COCTELES",
    "GASTROBAR",
    "TAPAS",
]

EPIGRAFES_HOSTELERIA = [
    "6311",  # Bares
    "5610",  # Restaurantes
    "5630",  # Cafeterías
    "5520",  # Cantinas
]

# =============================================================================
# LOCALIDAD
# =============================================================================
DISTRITOS_MADRID = {
    'CENTRO', 'ARGANZUELA', 'RETIRO', 'SALAMANCA', 'CHAMBERÍ', 
    'TETUÁN', 'CHAMARTÍN', 'FUENCARRAL-EL PARDO', 'MONCLOA-ARAVACA', 
    'LATINA', 'CARABANCHEL', 'USERA', 'PUENTE DE VALLECAS', 'MORATALAZ', 
    'CIUDAD LINEAL', 'HORTALEZA', 'VILLAVERDE', 'VILLA DE VALLECAS', 
    'VICÁLVARO', 'SAN BLAS-CANILLEJAS', 'BARAJAS'
}



# Mapa completo de variables para la interfaz del usuario
ALIAS_VARIABLES = {
    "n_bares_202206": "Nº de Bares (2022)",
    "n_bares_202606": "Nº de Bares (2026)",
    "velocidad_crecimiento_anual": "Velocidad de Crecimiento Anual",
    "aceleracion_crecimiento": "Aceleración del Crecimiento",
    "tendencia_54m": "Tendencia Histórica (54 Meses)",
    "variabilidad_bares": "Variabilidad Comercial (Bares)",
    "diversidad_categorias": "Diversidad de Categorías Comerciales",
    "bares_por_1000hab": "Bares por 1.000 Habitantes",
    "poblacion_total": "Población Total",
    "edad_media": "Edad Media (Años)",
    "pct_extranjeros": "% Población Extranjera",
    "tasa_natalidad": "Tasa de Natalidad",
    "indice_diversidad_cultural": "Índice de Diversidad Cultural",
    "pct_menores_30": "% Población Menor de 30 Años",
    "renta_media_2023": "Renta Media (2023)",
    "renta_mediana_2023": "Renta Mediana (2023)",
    "cambio_renta_2022_2026": "Variación Renta (2022-2026)",
    "desigualdad_gini": "Índice de Desigualdad Gini",
    "pct_poblacion_renta_baja": "% Población Renta Baja",
    "distancia_centro": "Distancia al Centro (km)",
    "proximidad_estaciones_metro": "Proximidad a Estaciones de Metro",
    "precio_m2_registradores": "Precio m² (Registradores)",
    "distancia_metro_km": "Distancia al Metro (km)",
    "tiempo_pie_a_metro_min": "Tiempo a Pie al Metro (min)",
    "accesibilidad_metro": "Índice de Accesibilidad a Metro",
    "trend_score": "Score de Tendencia (Trend Score)"
}

def get_variable_label(col_name: str) -> str:
    """Devuelve la etiqueta limpia si existe en el mapeo; si no, formatea el texto base."""
    return ALIAS_VARIABLES.get(col_name, str(col_name).replace('_', ' ').title())
# =============================================================================
# LOGGING
# =============================================================================

LOG_FILE = LOGS_DIR / "proyecto.log"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logging():
    """Configurar logging."""
    # La consola de Windows suele usar cp1252, que no puede codificar
    # caracteres como '→'; reconfigurarla a UTF-8 evita que el logging
    # crashee al loguear esos caracteres.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()