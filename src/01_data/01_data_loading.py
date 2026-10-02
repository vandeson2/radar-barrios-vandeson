from pathlib import Path
from typing import Dict
import gc
import pandas as pd
import sys

current_dir = Path(__file__).resolve()
for parent in current_dir.parents:
    if (parent / "config.py").exists():
        sys.path.insert(0, str(parent))
        break

from config import (
    ANIOS_ESTRUCTURA,
    ANIOS_PROCESAR,
    NOMBRES_MESES,
    PROCESSED_PATHS,
    CSV_READ_OPTIONS,
    DATOS_COMPLEMENTARIOS,
    logger,
)


ENCODINGS_FALLBACK = ["cp1252", "latin-1"]


def leer_csv_robusto(archivo: Path) -> pd.DataFrame:
    """
    Lee un CSV con CSV_ENCODING y, si falla por encoding, reintenta
    con encodings alternativos (algunos archivos no son UTF-8 real).
    """
    try:
        return pd.read_csv(archivo, **CSV_READ_OPTIONS)
    except UnicodeDecodeError:
        opciones = dict(CSV_READ_OPTIONS)
        for encoding in ENCODINGS_FALLBACK:
            opciones["encoding"] = encoding
            try:
                df = pd.read_csv(archivo, **opciones)
                logger.warning(f"{archivo.name}: leído con encoding alternativo '{encoding}'")
                return df
            except UnicodeDecodeError:
                continue
        raise


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Quita caracteres residuales de BOM (ej. '?') pegados al primer header."""
    df.columns = df.columns.str.lstrip("?﻿").str.strip()
    return df


# Columnas de metadata de ubicación/edificio que vienen repetidas EN AMBOS
# archivos (locales y actividades) con el mismo valor para un mismo local_id.
# Verificado empíricamente (2022-10, 2024-06, 2025-04): coincidencia 100%.
# Se descartan del lado de actividades antes del merge para no duplicar
# ~40 columnas (evita el MemoryError al consolidar los 5 años).
# 'fx_carga' se excluye a propósito: es el timestamp de extracción de cada
# archivo y sí difiere legítimamente entre locales y actividades.
COLUMNAS_LOCALES_DUPLICADAS_EN_ACTIVIDADES = [
    "id_distrito_local", "desc_distrito_local", "id_barrio_local",
    "desc_barrio_local", "cod_barrio_local", "id_seccion_censal_local",
    "desc_seccion_censal_local", "coordenada_x_local", "coordenada_y_local",
    "id_tipo_acceso_local", "desc_tipo_acceso_local", "id_situacion_local",
    "desc_situacion_local", "id_vial_edificio", "clase_vial_edificio",
    "desc_vial_edificio", "id_ndp_edificio", "id_clase_ndp_edificio",
    "nom_edificio", "num_edificio", "cal_edificio", "secuencial_local_PC",
    "id_vial_acceso", "clase_vial_acceso", "desc_vial_acceso",
    "id_ndp_acceso", "id_clase_ndp_acceso", "nom_acceso", "num_acceso",
    "cal_acceso", "coordenada_x_agrupacion", "coordenada_y_agrupacion",
    "id_agrupacion", "nombre_agrupacion", "id_tipo_agrup", "desc_tipo_agrup",
    "id_planta_agrupado", "id_local_agrupado", "rotulo", "fx_datos_fin",
]


# El esquema real de "locales" varió con el tiempo (ej. cod_postal y las
# horas de apertura no existían antes de oct-2022), así que no se puede
# validar por columnas requeridas. En cambio, se detectan archivos MAL
# ETIQUETADOS: ej. 202202_Febrero_locales.csv en realidad es un archivo de
# TERRAZAS (columnas id_terraza, mesas_es, ...) con 6,915 filas en vez de
# las ~167,000 esperadas para ese mes.
COLUMNAS_MARCADORAS_TERRAZA = {"id_terraza", "mesas_es", "sillas_es", "id_periodo_terraza"}
COLUMNAS_ESPERADAS_ACTIVIDADES = {"id_epigrafe", "desc_epigrafe"}


def validar_estructura_locales(df: pd.DataFrame, archivo: Path) -> bool:
    """Rechaza archivos de terrazas guardados con nombre de 'locales'."""
    marcadores = COLUMNAS_MARCADORAS_TERRAZA & set(df.columns)
    if marcadores:
        logger.error(
            f"{archivo.name}: contiene columnas de terrazas {sorted(marcadores)} - "
            f"es un archivo de terrazas mal nombrado como 'locales', se descarta este mes"
        )
        return False
    if "id_local" not in df.columns:
        logger.error(f"{archivo.name}: no tiene columna 'id_local', se descarta este mes")
        return False
    return True


def validar_estructura(df: pd.DataFrame, columnas_esperadas: set, archivo: Path) -> bool:
    """Verifica que el CSV tenga las columnas del censo real de actividades."""
    faltantes = columnas_esperadas - set(df.columns)
    if faltantes:
        logger.error(
            f"{archivo.name}: no tiene la estructura esperada (faltan columnas "
            f"{sorted(faltantes)}) - probablemente es un archivo de otro tipo mal "
            f"nombrado, se descarta este mes"
        )
        return False
    return True


def normalizar_tipos_para_parquet(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.select_dtypes(include="object", exclude="str").columns:
        no_nulos = df[col].notna()
        if no_nulos.sum() == 0:
            continue
        numerico = pd.to_numeric(df[col], errors="coerce")
        if (numerico.notna() == no_nulos).all():
            df[col] = numerico
        else:
            df[col] = df[col].astype("string")
    return df


def buscar_archivos(directorio: Path, anio: int, mes: int, tipo:str) -> Path:
    """
    Busca archivos CASV con patrón flexible.

    Args:
        directorio: donde buscar
        anio: año (2022, 2023 ...)
        mes: mes (1-12)
        tipo: "actividade" o "locales"

    Returns:
        Path al archivo encontrado o None
    """

    patron = f"{anio}{mes:02d}*{tipo}*.csv"
    archivos = list(directorio.glob(patron))

    if not archivos:
        return None

    if len(archivos) > 1:
        logger.warning(f"Múltiples aechivos para {anio}-{mes:02d} ({tipo}): {[f.name for f in archivos]}")
        return archivos[0] # se queda con el primero

    return archivos [0]


def carga_mes(anio: int, mes: int) -> pd.DataFrame:
    """
    Carga un mes esècífico (Actividad + locales)

    Arg:
       anio: año (2022, 2023 ...)
        mes: mes (1-12)
    
    Returns:
        DataFrame fusionado o None si falta archivos
    """

    directorio = ANIOS_ESTRUCTURA[anio]["directorio"]
    mes_nombre = NOMBRES_MESES[mes]

    #Buscar archivo
    archivo_act = buscar_archivos(directorio, anio, mes, "actividades")
    archivo_loc = buscar_archivos(directorio, anio, mes, "locales")

    if not archivo_act or not archivo_loc:
        logger.warning(f" - {anio}-{mes}: Faltan archivos")
        if not archivo_act:
            logger.warning("  Falta: Actividades")
        if not archivo_loc:
            logger.warning("  Falta: Locales")
        return None

    try:
        # Carga CSV 
        df_act = leer_csv_robusto(archivo_act)
        df_loc = leer_csv_robusto(archivo_loc)

        # Normalizar nombres de columnas (algunos archivos traen un BOM/caracter
        # residual pegado al primer header, ej. "?id_local" en vez de "id_local")
        df_act = normalizar_columnas(df_act)
        df_loc = normalizar_columnas(df_loc)

        # Descartar archivos mal etiquetados (ej. archivo de terrazas
        # guardado con nombre de "locales")
        if not validar_estructura(df_act, COLUMNAS_ESPERADAS_ACTIVIDADES, archivo_act):
            return None
        if not validar_estructura_locales(df_loc, archivo_loc):
            return None

        # Crear columna fecha
        df_act['fecha'] = f"{anio}{mes:02d}"
        df_loc['fecha'] = f"{anio}{mes:02d}"

        # Renombrar columnas para compatibilidad
        df_act = df_act.rename(columns={
            'id_local': 'local_id',
            'fx_datos_ini': 'fecha_apertura',
            'desc_epigrafe': 'descripcion_actividad'
        })
        df_loc = df_loc.rename(columns={'id_local': 'local_id'})

        # Descartar del lado de actividades las columnas de ubicación/edificio
        # que ya vienen en df_loc con el mismo valor (evita duplicarlas _x/_y)
        columnas_a_descartar = [
            c for c in COLUMNAS_LOCALES_DUPLICADAS_EN_ACTIVIDADES if c in df_act.columns
        ]
        df_act = df_act.drop(columns=columnas_a_descartar)

        logger.info(f"{anio}-{mes:02d} ({mes_nombre}): {df_act.shape[0]:,} + {df_loc.shape[0]:,}")

        # Fusionar por (local_id, fecha)
        df_mes = df_loc.merge(
            df_act,
            on=['local_id', 'fecha'],
            how='left'
        )

        return df_mes

    except Exception as e:
        logger.error(f"Error cargando {anio}-{mes:02d}: {e}")
        return None


def carga_anio (anio: int) -> pd.DataFrame:
    """
    Carga un año completo

    Args:
        anio: año a cargar (2022, 2023...)
    
    Returns:
        DataFrame con todos los meses del año
    """

    logger.info(f"\n CARGANDO AÑO {anio}")
    logger.info("-" * 60)

    if anio not in ANIOS_ESTRUCTURA:
        logger.error(f"Año {anio} no encontrado")
        return None

    meses = ANIOS_ESTRUCTURA[anio]["meses"]
    dfs_anio = []

    for mes in meses:
        df_mes = carga_mes(anio, mes)
        if df_mes is not None and len(df_mes) > 0:
            dfs_anio.append(df_mes)

    if not dfs_anio:
        logger.error(f"No se ha cargó ningún mes de {anio}")
        return None

    #Consolidar año
    df_anio = pd.concat(dfs_anio, ignore_index=True)
    df_anio = normalizar_tipos_para_parquet(df_anio)
    logger.info(f"Año {anio}: {df_anio.shape[0]:,} registros totales")

    return df_anio


def cargar_todos_anios() -> Dict:
    """
    Carga cada año, lo guarda en su propio parquet (PROCESSED_PATHS
    'consolidated_{anio}') y libera la memoria antes de pasar al siguiente.

    Returns:
        Dict con el resumen de la carga
    """

    logger.info("\n" + "=" * 80)
    logger.info("INICIANDO CARGA DE DATOS MULTIYEAR")
    logger.info("=" * 80)
    logger.info(f"Años a procesar: {ANIOS_PROCESAR}")
    logger.info("=" * 80 )

    PROCESSED_PATHS['consolidated'].parent.mkdir(parents=True, exist_ok=True)

    anios_cargados = []
    anios_fallidos = []
    registros_por_anio = {}
    fechas_min = []
    fechas_max = []
    barrios_unicos = set()

    for anio in ANIOS_PROCESAR:
        df_anio = carga_anio(anio)

        if df_anio is None or len(df_anio) == 0:
            anios_fallidos.append(anio)
            continue

        output_path = PROCESSED_PATHS.get(f"consolidated_{anio}")
        if output_path is None:
            logger.warning(f"No hay ruta configurada para guardar el año {anio}")
        else:
            df_anio.to_parquet(output_path, index=False)
            logger.info(
                f"Guardado {anio}: {output_path.name} "
                f"({output_path.stat().st_size / (1024**2):.2f} MB)"
            )

        anios_cargados.append(anio)
        registros_por_anio[anio] = len(df_anio)
        fechas_min.append(df_anio['fecha'].min())
        fechas_max.append(df_anio['fecha'].max())
        barrios_unicos.update(df_anio['desc_barrio_local'].dropna().unique())

        # Liberar el año antes de cargar el siguiente
        del df_anio
        gc.collect()

    if not anios_cargados:
        logger.error("No se cargó ningún año")
        return None

    total_registros = sum(registros_por_anio.values())

    logger.info("\n" + "=" * 80)
    logger.info("RESUMEN DE CARGA")
    logger.info("=" * 80)
    logger.info(f"Años cargados: {anios_cargados}")
    if anios_fallidos:
        logger.warning(f"Años fallidos: {anios_fallidos}")
    for anio, n in registros_por_anio.items():
        logger.info(f"  {anio}: {n:,} registros")
    logger.info(f"Total registros: {total_registros:,}")
    logger.info(f"Rango de fechas: {min(fechas_min)} → {max(fechas_max)}")
    logger.info(f"Barrios únicos: {len(barrios_unicos)}")
    logger.info("=" * 80 + "\n")

    return {
        "anios_cargados": anios_cargados,
        "anios_fallidos": anios_fallidos,
        "registros_por_anio": registros_por_anio,
        "total_registros": total_registros,
    }






# =============================================================================
# CARGA OTROS DATASETS
# =============================================================================
def cargar_padron() -> pd.DataFrame:
    """Cargar Padrón Municipal."""
    logger.info("\n Cargando Padrón Municipal...")
    
    try:
        path = DATOS_COMPLEMENTARIOS["padron"]["path"]
        
        if not path.exists():
            logger.warning(f"   Archivo no encontrado: {path}")
            return None
        
        df = pd.read_csv(
            path,
            delimiter=DATOS_COMPLEMENTARIOS["padron"]["delimiter"],
            encoding=DATOS_COMPLEMENTARIOS["padron"]["encoding"]
        )
        
        # Limpiar nombres de columnas 
        df.columns = [col.strip().strip('"') for col in df.columns]
        
        logger.info(f"     Cargado: {len(df):,} filas × {df.shape[1]} columnas")
        logger.info(f"     Distritos: {df['DESC_DISTRITO'].nunique()}")
        logger.info(f"     Barrios: {df['DESC_BARRIO'].nunique()}")
        
        return df
        
    except Exception as e:
        logger.error(f"  Error: {e}")
        return None


def cargar_renta() -> pd.DataFrame:
    """Cargar Indicadores de Renta (filtrado para Madrid)."""
    logger.info("\n Cargando Indicadores de Renta...")
    
    try:
        path = DATOS_COMPLEMENTARIOS["renta"]["path"]
        
        if not path.exists():
            logger.warning(f"  Archivo no encontrado: {path}")
            return None
        
        logger.info(f"  Cargando {path.stat().st_size / 1e6:.0f} MB...")
        
        df = pd.read_csv(
            path,
            delimiter=DATOS_COMPLEMENTARIOS["renta"]["delimiter"],
            encoding=DATOS_COMPLEMENTARIOS["renta"]["encoding"],
            on_bad_lines='skip'
        )
        
        logger.info(f"  Cargado total: {len(df):,} filas")
        
        # Filtrar solo Madrid
        df_madrid = df[df['Municipios'].str.contains('Madrid', case=False, na=False)].copy()
        logger.info(f"     Madrid filtrado: {len(df_madrid):,} filas")
        logger.info(f"     Años: {sorted(df_madrid['Periodo'].unique())}")
        logger.info(f"     Indicadores: {df_madrid['Indicadores de renta media'].nunique()}")
        
        return df_madrid
        
    except Exception as e:
        logger.error(f"  Error: {e}")
        return None


def cargar_precios() -> pd.DataFrame:
    """Cargar Precios Históricos."""
    logger.info("\n Cargando Precios Históricos...")
    
    try:
        path = DATOS_COMPLEMENTARIOS["precios"]["path"]
        
        if not path.exists():
            logger.warning(f"  Archivo no encontrado: {path}")
            return None
        
        df = pd.read_csv(
            path,
            delimiter=DATOS_COMPLEMENTARIOS["precios"]["delimiter"],
            encoding=DATOS_COMPLEMENTARIOS["precios"]["encoding"]
        )
        
        logger.info(f"     Cargado: {len(df):,} filas × {df.shape[1]} columnas")
        logger.info(f"     Distritos: {df['distrito'].nunique()}")
        logger.info(f"     Años: {sorted(df['año'].unique())}")
        
        return df
        
    except Exception as e:
        logger.error(f"  Error: {e}")
        return None


def cargar_barrios() -> pd.DataFrame:
    """Cargar Barrios."""
    logger.info("\n Cargando Barrios...")
    
    try:
        path = DATOS_COMPLEMENTARIOS["barrios"]["path"]
        
        if not path.exists():
            logger.warning(f"   Archivo no encontrado: {path}")
            return None
        
        df = pd.read_csv(
            path,
            delimiter=DATOS_COMPLEMENTARIOS["barrios"]["delimiter"],
            encoding=DATOS_COMPLEMENTARIOS["barrios"]["encoding"]
        )
        
        # Convertir Area de string a float
        df['Area'] = df['Area'].str.replace(',', '.').astype(float)
        
        logger.info(f"     Cargado: {len(df):,} filas × {df.shape[1]} columnas")
        logger.info(f"     Distritos: {df['CODDIS'].nunique()}")
        logger.info(f"     Barrios: {df['COD_BAR'].nunique()}")
        logger.info(f"     Área total: {df['Area'].sum() / 1e6:.1f} km²")
        
        return df
        
    except Exception as e:
        logger.error(f"  Error: {e}")
        return None

#--------------------------------------------------------------------
def cargar_datos_complementarios() -> Dict[str, pd.DataFrame]:
    """Cargar todos los datos complementarios."""
    logger.info(" Cargando datos complementarios...")
    
    datos = {}
    
    df_padron = cargar_padron()
    if df_padron is not None:
        datos["padron"] = df_padron
    
    df_renta = cargar_renta()
    if df_renta is not None:
        datos["renta"] = df_renta
    
    df_precios = cargar_precios()
    if df_precios is not None:
        datos["precios"] = df_precios
    
    df_barrios = cargar_barrios()
    if df_barrios is not None:
        datos["barrios"] = df_barrios
    
    logger.info("\n" + "=" * 80)
    logger.info(f" Datos complementarios cargados: {len(datos)}/4")
    logger.info("=" * 80 + "\n")
    
    return datos

    
def verificar_estructura():
    """
    Verifica que todos los archos esperados existan.

    Returns:
        Dict con estado de cada año/mes
    """

    logger.info("\n" + "=" * 80)
    logger.info("VERIFICA ESTRUCTURA DE ARCHIVOS")
    logger.info("=" * 80)

    verificacion = {}
    total_esperado = 0
    total_encontrado = 0

    for anio in ANIOS_PROCESAR:
        config = ANIOS_ESTRUCTURA[anio]
        directorio = config["directorio"]
        meses = config["meses"]

        logger.info(f"\n{anio}")
        verificacion[anio] = {"meses": {}}

        for mes in meses:
            archivo_act = buscar_archivos(directorio, anio, mes, "actividades")
            archivo_loc = buscar_archivos(directorio, anio, mes, "locales")

            total_esperado += 2
            tiene_act = archivo_act is not None
            tiene_loc = archivo_loc is not None

            if tiene_act and tiene_loc:
                logger.info(f"{anio}-{mes:02d}")
                total_encontrado += 2
                verificacion[anio]["meses"][mes]= True
            else:
                status = []
                if not tiene_act:
                    status.append("falta Actividades")
                if not tiene_loc:
                    status.append("falta Locales")
                logger.warning(f"{anio}-{mes:02d} ({', '.join(status)})")
                verificacion[anio]["meses"][mes] = False
                if tiene_act:
                    total_encontrado += 1
                if tiene_loc:
                    total_encontrado += 1

    # Verificar datos complementarios
    logger.info(f"\n DATOS COMPLEMENTARIOS:")
    for nombre, config in DATOS_COMPLEMENTARIOS.items():
        path = config["path"]
        existe = path.exists()
        status = "Cargados" if existe else "Error"
        tamaño = f"({path.stat().st_size / 1e6:.1f} MB)" if existe else ""
        logger.info(f"  {status} {nombre}: {path.name} {tamaño}")
    

    logger.info("\n" + "=" * 80)
    logger.info(f"RESUMEN: {total_encontrado}/{total_esperado} archivos encontrados")
    logger.info("=" * 80 + "\n")
    
    return verificacion


def guardar_datasets( datos_complementarios: Dict[str, pd.DataFrame]):
    """Guardar todos los datasets en parquet."""
    logger.info(" Guardando datasets..")

    # Guardar complementarios
    for nombre, df in datos_complementarios.items():
        if df is not None:
            output_path = PROCESSED_PATHS['consolidated'].parent / f"dataset_{nombre}.parquet"
            df.to_parquet(output_path, index=False, compression='snappy')
            logger.info(f" {nombre.capitalize()}: {output_path.name} ({len(df):,} registros)")
    
    logger.info("=" * 80 + "\n")
# =============================================================================
# MAIN
# =============================================================================
def main():
    """ Función principal"""

    #Verificar estructura
    verificacion = verificar_estructura()

    #Cargar y guardar datos (un parquet por año, sin consolidar todo en RAM)
    resumen = cargar_todos_anios()
    if resumen is None or resumen["total_registros"] == 0:
        logger.error("No se pudo cargar ningún dato")
        return False
    datos_complementarios = cargar_datos_complementarios()
    if not datos_complementarios:
        logger.warning(" No se cargaron los datos complementarios")
    else:
        guardar_datasets(datos_complementarios)

    logger.info("\n" + "=" * 80)
    logger.info("CARGA COMPLETADA EXITOSAMENTE")
    logger.info(
        f"{len(resumen['anios_cargados'])} años guardados en "
        f"{PROCESSED_PATHS['consolidated'].parent}"
    )
    logger.info("=" * 80)

    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"\n ERROR: {e}", exc_info=True)
        exit(1)