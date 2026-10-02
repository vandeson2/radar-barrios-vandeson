import importlib.util
from pathlib import Path
import logging
import sys
import time
import pandas as pd

current_dir = Path(__file__).resolve()
for parent in current_dir.parents:
    if (parent / "config.py").exists():
        sys.path.insert(0, str(parent))
        break

from config import PATHS, logger


def cargar_funcion_modulo(ruta_script: str, nombre_funcion: str):
    """Carga dinámicamente una función desde un script de Python permitiendo

    carpetas o archivos con números al inicio en su nombre.
    """
    path_script = Path(ruta_script).resolve()
    if not path_script.exists():
        raise FileNotFoundError(
            f"No se encuentra el script requerido: {ruta_script}"
        )

    spec = importlib.util.spec_from_file_location(
        path_script.stem, path_script
    )
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)

    if not hasattr(modulo, nombre_funcion):
        raise AttributeError(
            f"La función '{nombre_funcion}' no existe en {ruta_script}"
        )

    return getattr(modulo, nombre_funcion)

def ejecutar_pipeline_enrichment(
    ruta_input_gold: Path | str = PATHS["gold"]["gold_1"],
    ruta_output_gold: Path | str = PATHS["gold"]["gold_final"],
):
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE LIMPIEZA Y ENRIQUECIMIENTO (04_ENRICHMENT)")
    logger.info("=" * 80)

    # ------------------------------------------------------------------------
    # PASO 1: CARGAR DATASET GENERADO POR _06_capa_gold
    # ------------------------------------------------------------------------
    logger.info("\n>>> PASO 1: CARGANDO DATASET BASE DE LA CAPA GOLD")

    path_input = Path(ruta_input_gold)

    if not path_input.exists():
        
        posibles_rutas = [
            Path("data/gold/gold_barrios_base.parquet"),
            Path("data/gold/gold_barrios.parquet"),
            Path("data/gold/gold_barrios_base.csv"),
            Path("data/gold/gold_barrios.csv"),
        ]
        for p in posibles_rutas:
            if p.exists():
                path_input = p
                break

    if not path_input.exists():
        logger.error(
            f"No se encontró el dataset base generado por `_06_capa_gold`. "
            f"Buscado en: {ruta_input_gold}"
        )
        sys.exit(1)

    try:
        if path_input.suffix == ".parquet":
            df_gold_base = pd.read_parquet(path_input)
        else:
            df_gold_base = pd.read_csv(path_input)

        logger.info(
            f"Dataset cargado desde '{path_input}': "
            f"{len(df_gold_base)} barrios × {len(df_gold_base.columns)} columnas"
        )
    except Exception as e:
        logger.error(
            f"Error al leer el dataset de entrada: {str(e)}", exc_info=True
        )
        sys.exit(1)

    # ------------------------------------------------------------------------
    # PASO 2: EJECUTAR MÓDULOS DE ENRIQUECIMIENTO (04_enrichment)
    # ------------------------------------------------------------------------
    logger.info("\n>>> PASO 2: APLICANDO SECUENCIA DE ENRIQUECIMIENTO")

    try:
        # Cargar funciones de los scripts de la carpeta 04_enrichment
        limpiar_capa_gold = cargar_funcion_modulo(
            "src/04_enrichment/_00_cleaning_gold.py", "limpiar_capa_gold"
        )
        enriquecer_coordenadas = cargar_funcion_modulo(
            "src/04_enrichment/_01_enriquecer_coordenadas.py",
            "enriquecer_coordenadas",
        )
        enriquecer_precios = cargar_funcion_modulo(
            "src/04_enrichment/_02_enriquecer_precios.py", "enriquecer_precios"
        )
        calcular_accesibilidad_metro = cargar_funcion_modulo(
            "src/04_enrichment/_03_enriquecer_distancia_metro.py",
            "calcular_accesibilidad_metro",
        )
        enriquecer_capa_gold = cargar_funcion_modulo(
            "src/04_enrichment/_04_enriquecer_gold.py", "enriquecer_capa_gold"
        )

        # 1. Limpieza de datos
        logger.info("1/5 Limpiando registros de la capa Gold base...")
        df_clean = limpiar_capa_gold(df_gold_base)

        # 2. Enriquecer Coordenadas
        logger.info("2/5 Enriqueciendo Coordenadas (Latitud/Longitud)...")
        df_cord_madrid = pd.read_csv(PATHS["data"]["barrios_Madrid"])
        df_coords = enriquecer_coordenadas(df_clean, df_cord_madrid)

        # 3. Enriquecer Precios
        logger.info("3/5 Obteniendo Precios de Registradores...")
        df_precios_madrid = pd.read_csv(PATHS["data"]["enriquecer_precios"])
        df_precios = enriquecer_precios(df_coords, df_precios_madrid)

        # 4. Enriquecer Metro
        logger.info("4/5 Calculando Distancias y Accesibilidad a Metro...")
        df_metro = calcular_accesibilidad_metro(df_coords)

        # 5. Enriquecimiento y Consolidación Final
        logger.info("5/5 Generando Enriquecimiento Final Gold...")
        df_gold_final = enriquecer_capa_gold(
            df_gold=df_coords, df_precios=df_precios, df_metro=df_metro
        )

        # --------------------------------------------------------------------
        # PERSISTENCIA EN DISCO
        # --------------------------------------------------------------------
        path_output = Path(ruta_output_gold)
        path_output.parent.mkdir(parents=True, exist_ok=True)

        if path_output.suffix == ".parquet":
            df_gold_final.to_parquet(path_output, index=False)
        else:
            df_gold_final.to_csv(path_output, index=False)

        logger.info(f"Dataset final guardado exitosamente en: {path_output}")

    except Exception as e:
        logger.error(
            f"Error crítico durante el enriquecimiento: {str(e)}",
            exc_info=True,
        )
        sys.exit(1)

    # ------------------------------------------------------------------------
    # RESUMEN Y FINALIZACIÓN
    # ------------------------------------------------------------------------
    elapsed_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info(f"Tiempo de ejecución: {elapsed_time:.2f} segundos")
    logger.info(
        f"Dimensiones finales: {len(df_gold_final)} barrios × {len(df_gold_final.columns)} columnas"
    )
    logger.info(
        f"Nulos totales en dataset final: {df_gold_final.isna().sum().sum()}"
    )
    logger.info("=" * 80)

    return df_gold_final


if __name__ == "__main__":
    ejecutar_pipeline_enrichment()