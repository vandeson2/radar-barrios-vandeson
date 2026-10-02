"""
Entrena ENSEMBLE con Calibración Sigmoide StratifiedKFold
"""

import json
import pickle
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Cargar config
current_dir = Path(__file__).resolve()
for parent in current_dir.parents:
    if (parent / "config.py").exists():
        sys.path.insert(0, str(parent))
        break

from config import PATHS, logger

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
gold_path = PATHS['gold']['gold_final']
df_gold = pd.read_parquet(gold_path)

feature_names_path = gold_path.parent.parent / '04_train_test' / 'feature_names_v2_mejorado.json'
with open(feature_names_path, 'r') as f:
    feature_names = json.load(f)

# Asegurar tipo numérico y rellenar nulos
X = df_gold[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0.0)
y = df_gold['gentrificara']

print(f"Dataset cargado: {X.shape[0]} barrios y {X.shape[1]} características.")
print(f"Distribución del target: {y.value_counts(normalize=True).to_dict()}")

# ============================================================================
# 2. ESCALAR
# ============================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================================
# 3. ENSEMBLE BASE (RESTAURADO PROBABILITY=TRUE)
# ============================================================================
svm_modelo = SVC(
    kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
)

rf_modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)

gb_modelo = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
)

ensemble_base = VotingClassifier(
    estimators=[('svm', svm_modelo), ('rf', rf_modelo), ('gb', gb_modelo)],
    voting='soft',
)

# ============================================================================
# 4. CALIBRACIÓN SIGMOIDE SUAVE
# ============================================================================

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

ensemble_calibrated = CalibratedClassifierCV(
    estimator=ensemble_base, method='sigmoid', cv=skf
)
ensemble_calibrated.fit(X_scaled, y)

# ============================================================================
# 5. EVALUACIÓN Y PREDICCIONES
# ============================================================================
probs = ensemble_calibrated.predict_proba(X_scaled)[:, 1]

# UMBRALES ABSOLUTOS DE NEGOCIO
UMBRAL_BAJO = 0.35
UMBRAL_ALTO = 0.65

bajo = (probs < UMBRAL_BAJO).sum()
medio = ((probs >= UMBRAL_BAJO) & (probs <= UMBRAL_ALTO)).sum()
alto = (probs > UMBRAL_ALTO).sum()

print('\n--- RESULTADOS DE LAS PREDICCIONES ---')
print(f' • Mínima: {probs.min():.4f} | Máxima: {probs.max():.4f}')
print(f' • Media:  {probs.mean():.4f} | Mediana: {np.median(probs):.4f}')
print(f' • Desviación Estándar: {probs.std():.4f}')

print(f'\nCategorización por Umbrales Absolutos:')
print(
    f' • BAJO  (<{UMBRAL_BAJO*100:.0f}%): {bajo} barrios ({100*bajo/len(probs):.1f}%)'
)
print(
    f' • MEDIO ({UMBRAL_BAJO*100:.0f}%-{UMBRAL_ALTO*100:.0f}%): {medio} barrios ({100*medio/len(probs):.1f}%)'
)
print(
    f' • ALTO  (>{UMBRAL_ALTO*100:.0f}%): {alto} barrios ({100*alto/len(probs):.1f}%)'
)

# ============================================================================
# 6. GUARDAR ARTEFACTOS
# ============================================================================
models_dir = feature_names_path.parent
models_dir.mkdir(parents=True, exist_ok=True)

scaler_path = models_dir / 'scaler_v2_mejorado.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

model_path = models_dir / 'modelo_ensemble_v2_mejorado.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(ensemble_calibrated, f)

print("\nArtefactos exportados correctamente:")
print(f"  • Modelo: {model_path}")
print(f"  • Scaler: {scaler_path}")
print(f"  • Features: {feature_names_path}")