"""
SCRIPT 0: GENERAR FEATURE NAMES
===============================

Extrae la lista de features numéricos del GOLD enriquecido
y los guarda en feature_names_v2_mejorado.json

Ejecución:
    python src/05_ml_training/generar_feature_names.py
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Cargar config
current_dir = Path(__file__).resolve()
for parent in current_dir.parents:
    if (parent / "config.py").exists():
        sys.path.insert(0, str(parent))
        break

from config import PATHS, logger

print("=" * 100)
print("PASO 0: GENERANDO FEATURE NAMES")
print("=" * 100)

# ============================================================================
# 1. CARGAR GOLD ENRIQUECIDO
# ============================================================================

print("\n1. Cargando GOLD enriquecido...\n")

try:
    gold_path = PATHS['gold']['gold_final']
    df_gold = pd.read_parquet(gold_path)
    print(f"GOLD: {len(df_gold)} barrios × {len(df_gold.columns)} columnas")
except FileNotFoundError:
    print(f"ERROR: No se encontró {gold_path}")
    print("   Ejecuta primero el pipeline de enrichment")
    exit(1)
except KeyError:
    print("ERROR: Ruta 'gold_final' no definida en config.py")
    exit(1)

# ============================================================================
# 2. SELECCIONAR FEATURES NUMÉRICAS
# ============================================================================

print("\n2. Seleccionando features numéricas...\n")

# Columnas a EXCLUIR (no son features)
cols_excluir = [
    'barrio_id',           # ID
    'barrio_nombre',       # Nombre
    'gentrificara',        # Target
    'categoria_renta',     # Categoría (no numérica)
    'zona',                # Zona (no numérica)
    'latitud',             # Coordenadas (usadas en mapa, no en ML)
    'longitud',            # Coordenadas (usadas en mapa, no en ML)
    'distrito'             # Distrito (no numérica)
]

# Obtener features numéricas
feature_names = []
for col in df_gold.columns:
    # Saltar columnas excluidas
    if col in cols_excluir:
        continue
    
    # Incluir solo numéricas
    if df_gold[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
        feature_names.append(col)

print(f"   Total de features numéricos: {len(feature_names)}\n")
print("   Features seleccionados:")
for i, feat in enumerate(feature_names, 1):
    print(f"   {i:2d}. {feat}")

# ============================================================================
# 3. VERIFICACIÓN
# ============================================================================

print("\n3. Verificación...\n")

# Verificar que no hay NaNs en features
missing = df_gold[feature_names].isnull().sum().sum()
print(f"   Datos faltantes en features: {missing}")

if missing > 0:
    print("Advertencia: Hay datos faltantes en features")
    print("   (Se rellenarán durante el entrenamiento)")

print(f"\nFeatures verificados: LISTO")

# ============================================================================
# 4. GUARDAR FEATURE NAMES
# ============================================================================

print("\n4. Guardando feature_names...\n")

output_path = PATHS['gold']['base'].parent.parent / '04_train_test' / 'feature_names_v2_mejorado.json'
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(feature_names, f, indent=2)

print(f"Guardado: {output_path}")

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "=" * 100)
print("RESUMEN")
print("=" * 100)

print(f"""
FEATURE NAMES GENERADO

Archivo:     {output_path}
Features:    {len(feature_names)}
Barrios:     {len(df_gold)}

Columnas EXCLUIDAS (no son features):
  - barrio_id, barrio_nombre (ID)
  - gentrificara (Target)
  - latitud, longitud (Coordenadas)
  - categoria_renta, zona, distrito (Categóricas)

STATUS: COMPLETADO
""")

print("=" * 100)
