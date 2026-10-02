import pandas as pd
from pathlib import Path
import sys

current_dir = Path(__file__).resolve()
for parent in current_dir.parents:
    if (parent / "config.py").exists():
        sys.path.insert(0, str(parent))
        break
from config import (PATHS, logger)
from _00_tabla_base import generar_base_barrios
from _01_feature_hosteleria import features_hosteleria
from _02_feature_demografico import features_demografico
from _03_feature_economico import features_economico
from _04_feature_geografico import features_geografica
from _05_target import generar_target

def generar_capa_gold(f_base, funciones_secundarias) -> pd.DataFrame:
    """
    Generar CAPA GOLD a partir de una función base y una lista/diccionario
    de funciones secundarias para fusionar por 'barrio_id'.

    Total: 128 barrios × 30 columnas
    """
    # Base inicial
    df_gold = f_base()
    logger.info(f"Obteniendo base inicial {df_gold.shape}")

    logger.info("Generando capa base..")
    for nombre, funcion in funciones_secundarias.items():
        df_gold = df_gold.merge(funcion(), on='barrio_id', how='left')
        logger.info(f" + {nombre}: {df_gold.shape}")

    print()
    return df_gold



if __name__ == "__main__":
    
    # Definimos las funciones secundarias con sus nombres
    componentes_extra = {
        "Hostelería": features_hosteleria,
        "Demográficos": features_demografico,
        "Económicos": features_economico,
        "Geográficos": features_geografica,
        "Target": lambda: generar_target()[['barrio_id', 'gentrificara']]
    }

    # Llamada pasando la función base y el resto de funciones
    df_gold_final = generar_capa_gold(
        f_base=generar_base_barrios, 
        funciones_secundarias=componentes_extra
    )

    # Guardar
    output_path = PATHS['gold']['gold_1']
    df_gold_final.to_parquet(output_path, index=False)
    logger.info(f"Capa Gold guardada correctamente en: {output_path}")




