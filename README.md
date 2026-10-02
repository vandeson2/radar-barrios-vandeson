# 🏘️ Radar de Barrio: Predictor de Gentrificación en Madrid

**Proyecto de Trabajo Final de Máster (TFM)**  
**Autor:** Vandeson Sena e Silva  
**Universidad:** EVOLVE Academy 
**Año:** 2026

---

## 📋 Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Inicio Rápido](#-inicio-rápido)
3. [Instalación](#instalación)
4. [Ejecución del Pipeline](#⚙️-ejecución-del-pipeline)
5. [Estructura del Proyecto](#estructura-del-proyecto)
6. [Datos Utilizados](#datos-utilizados)
7. [Resultados Esperados](#resultados-esperados)
8. [Limitaciones y Caveats](#limitaciones-y-caveats)
9. [Documentación Adicional](#-documentación-adicional)
10. [Referencias](#referencias)

---

## 📖 Descripción General

### Problema que Resuelve

Cada año en Madrid, **inversores, planificadores urbanos y ciudadanos toman decisiones de inversión sin información sistemática sobre cuáles barrios van a gentrificarse**. Las señales de gentrificación (explosión de bares modernos, llegada de población joven, cambios en comercio) emergen **meses antes** de que se reflejen en precios, pero nadie las conecta de forma rigurosa.

**Ejemplo real:** Malasaña pasó de €3,500/m² (2018) a €5,200/m² (2020), un aumento de **48% en 2 años**. Los que compraron en 2019 ganaron; los que esperaron demasiado, perdieron la oportunidad.

### Solución Planteada

**Radar de Barrio** es un **clasificador de Machine Learning** que predice si un barrio de Madrid gentrificará en los próximos **12-18 meses** basándose en:

- 📊 **54 meses de evolución comercial** (Feb 2022 - Jun 2026)
- 👥 **Contexto demográfico** (población, edad, diversidad)
- 💰 **Datos económicos** (renta media/mediana)
- 🌍 **Factores geográficos** (distancia centro, estaciones metro)
- 💵 **Validación con precios reales** (Colegio de Registradores/TINSA)

### Resultados Entregables

✅ **Clasificador ML Ensemble:** Predice SÍ/NO gentrificación con probabilidad (0-100%)  
✅ **Dashboard Interactivo:** Streamlit con mapa de Madrid coloreado  
✅ **Explicabilidad SHAP:** Top 3 factores por barrio  
✅ **Validación Rigurosa:** Correlación con precios oficiales (Registradores)  
✅ **Ranking TOP 10:** Barrios en riesgo inmediato  
✅ **Informes Exportables:** PDF profesional para presentar a inversores  

---

## 🚀 Inicio Rápido

**En 3 pasos:**

```bash
# 1. Crear entorno virtual
python -m venv venv && source venv/bin/activate  # Linux/Mac
python -m venv venv && venv\Scripts\activate    # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar pipeline completo 
python pipeline_maestro.py --mode all

# Dashboard estará en: http://localhost:8501
```

**¿Ya tienes datos procesados?**
```bash
# Solo entrenar modelo y dashboard 
python pipeline_maestro.py --mode train-app
```

**¿Solo quieres ver el dashboard?**
```bash
# Abrir app con modelo pre-entrenado
python pipeline_maestro.py --mode dashboard
```

---

## 🚀 Instalación

### Requisitos Previos

- **Python 3.8+**
- **Git**
- **RAM:** 8+ GB recomendado
- **Espacio disco:** ~15 GB (datos raw + processed)

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/radar-barrios.git
cd radar-barrios
```

### Paso 2: Crear Entorno Virtual

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Paso 3: Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 4: Descargar Datos

Los datos se descargan automáticamente en el pipeline. Para descargarlos manualmente:

```bash
# Crear estructura de carpetas
mkdir -p data/raw/{2022,2023,2024,2025,2026} data/raw/otros

# Los datos deben estar en:
# - data/raw/YYYY/ : Censo de Locales (archivos CSV mensuales)
# - data/raw/otros/ : Padrón, Renta, Precios, Barrios
```

**Fuentes de datos públicas:**
- 📊 Censo Locales: https://datos.madrid.es/dataset/209548-0-censo-locales-historico
- 👥 Padrón Municipal: https://datos.madrid.es/dataset/200076-2-padron-municipal-csv
- 💰 Renta INE: Instituto Nacional de Estadística
- 🏠 Precios Registradores: Colegio de Registradores
- 🗺️ Límites Barrios: https://datos.madrid.es/dataset/300496-0-barrios-madrid

---

## ⚙️ Ejecución del Pipeline

### 🚀 OPCIÓN RECOMENDADA: Pipeline Maestro Automatizado

El proyecto incluye un **`pipeline_maestro.py`** que orquesta automáticamente los 9 stages de ejecución:

```bash
# Pipeline completo (2-3 horas)
python pipeline_maestro.py --mode all
```

**Stages ejecutados automáticamente:**
1. ✅ Data Loading (54 meses raw)
2. ✅ Data Cleaning (limpieza, deduplicación, valores nulos)
3. ✅ Consolidación Hostelería (5 años de datos)
4. ✅ Mapeo Barrios (fuzzy matching)
5. ✅ Capa Gold (128 × 32 features, tabla de features bruta)
6. ✅ Enriquecimiento (coordenadas, precios, distancia metro)
7. ✅ Generación Feature Names (listado de feature names)
8. ✅ Entrenamiento ML (Ensemble: SVM + Random Forest + Gradient Boosting)
9. ✅ Dashboard Streamlit (http://localhost:8501)

**Ventajas:**
- Ejecuta automáticamente en orden correcto
- Validación automática de requisitos antes de cada stage
- Logging completo con timestamp (`logs/pipeline_maestro_YYYYMMDD_HHMMSS.log`)
- Detención automática si fallan stages críticos (1-5)
- Manejo robusto de errores y timeouts (10 min por stage)
- Reproducible desde cero
- Posibilidad de ejecutar stages parciales

---

### 🎯 Otros Modos de Ejecución

El `pipeline_maestro.py` soporta varios modos para ejecutar partes específicas:

**Solo hasta Capa Gold (stages 1-5):**
```bash
python pipeline_maestro.py --mode hasta --stage 5
# Tiempo: ~1.5 horas
# Resultado: data/gold/gold_barrios_*.parquet
# Nota: Detiene antes de enriquecimiento y ML
```

**Datos + Enriquecimiento (stages 1-6):**
```bash
python pipeline_maestro.py --mode hasta --stage 6
# Tiempo: ~2 horas
# Resultado: datos enriquecidos con precios, metro, coordenadas
```

**Solo ML + Dashboard (stages 6-9, requiere datos gold previos):**
```bash
python pipeline_maestro.py --mode train-app
# Tiempo: ~20 minutos
# Requisitos: 
#   - data/gold/gold_barrios_enriquecido.parquet
#   - data/04_train_test/ (si existen)
```

**Solo Dashboard (stage 9, requiere modelo entrenado):**
```bash
python pipeline_maestro.py --mode dashboard
# Abre: http://localhost:8501
# Requisitos: 
#   - data/04_train_test/modelo_ensemble_v2_mejorado.pkl
#   - data/04_train_test/scaler_v2_mejorado.pkl
```

**Ejecutar stage específico:**
```bash
python pipeline_maestro.py --mode custom --stage 8
# Ejecuta solo el entrenamiento (stage 8)
```

**Limpiar caché de Streamlit:**
```bash
python pipeline_maestro.py --clean
# Elimina: ~/.streamlit, .pytest_cache/, __pycache__/
# Útil si hay problemas de caché con la app
```

---

### 📖 Cómo Funciona pipeline_maestro.py

El script orquestador ejecuta los siguientes pasos en orden:

```
pipeline_maestro.py
├─ Valida requisitos (config.py, src/, data/)
├─ Crea logs/pipeline_maestro_YYYYMMDD_HHMMSS.log
└─ Para cada stage (1-9):
   ├─ Verifica que exista el script
   ├─ Ejecuta con timeout de 10 minutos
   ├─ Captura stdout/stderr
   ├─ Registra resultado (OK/ERROR/TIMEOUT)
   └─ Detiene si stage crítico (1-5) falla
```

**Flujo de datos:**
```
Stage 1: Datos raw (CSV, 54 meses)
   ↓ (9M registros)
Stage 2: Datos limpios
   ↓ (sin nulos/duplicados)
Stages 3-4: Features por categoría
   ↓ (consolidados por barrio)
Stage 5: Tabla capa gold (128×32)
   ↓ (features brutos)
Stage 6: Capa gold enriquecida (precios, metro, coords)
   ↓
Stages 7-8: Modelo ML entrenado + importancia de features
   ↓
Stage 9: Dashboard interactivo (Streamlit)
```

**Archivos de Log:**
- Ubicación: `logs/pipeline_maestro_YYYYMMDD_HHMMSS.log`
- Contiene: timestamps, duración de cada stage, errores detallados
- Ver último log: `tail -f logs/pipeline_maestro_*.log`

---

### 📖 Ejecución Manual (Alternativa a pipeline_maestro.py)

Si prefieres ejecutar scripts individuales manualmente:

```bash
# 1. Cargar datos raw (54 meses, ~9M registros)
python src/01_data/01_data_loading.py
# Output: data/processed/01_consolidated/*.parquet

# 2. Limpiar datos (valores nulos, duplicados, outliers)
python src/02_cleaning/03_cleaning_main.py
# Output: data/processed/02_cleaned/*.parquet

# 3. Consolidar hostelería (agrupar por mes y barrio)
python src/03_feature/04_consolidar_hosteleria.py
# Output: data/processed/03_engineered/consolidado_hosteleria.parquet

# 4. Mapeo de barrios con fuzzy matching
python src/03_feature/mapeo/03_pipeline_mapeo_barrios.py
# Output: data/processed/03_engineered/mapping_resultados.csv

# 5. Capa Gold: Ingeniería de features (128 barrios × 32 features)
python src/03_feature/_06_capa_gold.py
# Output: data/gold/gold_barrios_*.parquet

# 6. Enriquecimiento: Añade coordenadas, precios, distancia metro
python src/04_enrichment/_05_pipeline_enriquecimiento.py
# Output: data/gold/gold_barrios_enriquecido.parquet

# 7. Generar listado de nombres de features
python src/05_ml_training/generar_feature_names.py
# Output: data/04_train_test/feature_names_v2_mejorado.json

# 8. Entrenar modelo Ensemble (SVM + RF + Gradient Boosting)
python src/05_ml_training/train.py
# Outputs: 
#   - data/04_train_test/modelo_ensemble_v2_mejorado.pkl
#   - data/04_train_test/scaler_v2_mejorado.pkl
#   - reports/feature_importance.png

# 9. Ejecutar Dashboard Streamlit
streamlit run src/06_dashboard/app.py
# Abre: http://localhost:8501
```

⚠️ **Nota:** La ejecución manual requiere respetar el orden exacto. El `pipeline_maestro.py` automatiza esto.

---

### 📊 Monitoreo y Logs del Pipeline

**Ver logs de ejecución:**

```bash
# Ver archivo de log más reciente
type logs/pipeline_maestro_*.log  # Windows
cat logs/pipeline_maestro_*.log   # Linux/Mac

# Ver últimas 50 líneas (útil mientras se ejecuta)
Get-Content logs/pipeline_maestro_*.log -Tail 50  # PowerShell
tail -50 logs/pipeline_maestro_*.log               # Bash

# Buscar errores en los logs
Select-String "ERROR" logs/pipeline_maestro_*.log  # PowerShell
grep "ERROR" logs/pipeline_maestro_*.log           # Bash
```

**Ejemplo de log:**
```
[15:59:53] INFO - ================================================================================
[15:59:53] INFO - PIPELINE MAESTRO - RADAR DE BARRIO
[15:59:53] INFO - ================================================================================
[16:00:01] INFO - Stage 1: Cargar datos raw (54 meses)
[16:00:01] INFO - Script: 01_data_loading.py
[16:02:15] SUCCESS - 01_data_loading - COMPLETADO
[16:02:15] INFO - Stage 2: Limpiar datos (valores nulos, duplicados)
...
[16:25:40] SUCCESS - PIPELINE COMPLETADO EXITOSAMENTE
[16:25:40] INFO - Duración total: 25.6 minutos
```

**En caso de error:**
```bash
# Ver el stage que falló
Select-String "ERROR|TIMEOUT" logs/pipeline_maestro_*.log

# Ver stderr del script que falló (últimos 300 caracteres)
Select-String "STDERR" logs/pipeline_maestro_*.log
```


## 📁 Estructura del Proyecto

```
radar-barrios/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python
├── config.py                          # Configuración centralizada
│
├── docs/entregas/                     # Directrices académicas
│   ├── 01_ideas_producto.md
│   ├── 02_datos_necesarios.md
│   ├── 03_modelo_datos.md
│   ├── 04_analisis_modelado.md
│   └── 05_diseño_frontal.md
│
├── src/                               # Código fuente (pipeline + dashboard)
│   ├── 01_data/                       # Carga de datos raw
│   │   ├── 01_data_loading.py         # Consolidación de 54 meses
│   │   └── harmonizar_columnas.py
│   │
│   ├── 02_cleaning/                   # Limpieza y normalización
│   │   ├── 03_cleaning_main.py
│   │   ├── cleaning_multiyear.py
│   │   └── cleaning_complementarios.py
│   │
│   ├── 03_feature/                    # Ingeniería de features
│   │   ├── _00_tabla_base.py          # Base: 128 barrios
│   │   ├── _01_feature_hosteleria.py  # 10 features (velocidad, aceleración, etc)
│   │   ├── _02_feature_demografico.py # 8 features (población, edad, etc)
│   │   ├── _03_feature_economico.py   # 6 features (renta, desigualdad, etc)
│   │   ├── _04_feature_geografico.py  # 3 features (distancia, metro, zona)
│   │   ├── _05_target.py              # Etiquetado manual (gentrificara: 0/1)
│   │   ├── _06_capa_gold.py           # Tabla final: 128×32 columnas
│   │   └── mapeo/                     # Mapeo barrios fuzzy
│   │
│   ├── 04_enrichment/                 # Enriquecimiento de datos
│   │   ├── _00_cleaning_gold.py
│   │   ├── _01_enriquecer_coordenadas.py
│   │   ├── _02_enriquecer_precios.py
│   │   ├── _03_enriquecer_distancia_metro.py
│   │   ├── _04_enriquecer_gold.py
│   │   └── _05_pipeline_enriquecimiento.py
│   │
│   ├── 05_ml_training/                # Entrenamiento y evaluación
│   │   ├── train.py                   # Versión final (Ensemble)
│   │   ├── train_v1_baseline.py
│   │   ├── generar_feature_names.py
│   │   └── 5_optimizar_mejorado_final.py
│   │
│   ├── 06_dashboard/                  # Frontend Streamlit
│   │   └── pr.py                      # Dashboard interactivo (+1400 líneas)
│   │
│   └── visualization/                 # Visualizaciones
│       └── __init__.py
│
├── notebooks/                         # Análisis exploratorio (16 notebooks)
│   ├── 01_eda_exploratory.ipynb
│   ├── 02_eda_after_cleaned.ipynb
│   ├── 03_Feature_Importance_SHAP.ipynb
│   ├── 04_eda_explo_barrios.ipynb
│   └── ... (más análisis)
│
├── data/                              # Datos en capas
│   ├── raw/                           # Raw sin procesar (~10 GB)
│   │   ├── 2022-2026/                 # 54 meses mensuales
│   │   └── otros/                     # Padrón, Renta, Precios, Barrios
│   │
│   ├── processed/                     # Datos limpios y procesados (~3 GB)
│   │   ├── 01_consolidated/
│   │   ├── 02_cleaned/
│   │   ├── 03_engineered/
│   │   └── 04_train_test/
│   │
│   └── gold/                          # Datos finales (capa gold)
│       ├── gold_barrios_completo.parquet (128×32)
│       ├── gold_barrios_predicciones.csv
│       └── gold_validacion_registradores.csv
│
├── models/                            # Modelos entrenados
│   ├── logistic_regression.pkl
│   ├── svm_model.pkl
│   ├── xgboost_model.pkl
│   ├── modelo_ensemble_v2_mejorado.pkl
│   └── scaler.pkl
│
├── logs/                              # Logs de ejecución
│   └── pipeline.log
│
└── reports/                           # Reportes y análisis
    ├── feature_importance.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## 📊 Datos Utilizados

### Fuentes Principales

| Fuente | Cobertura | Granularidad | Registros |
|--------|-----------|--------------|-----------|
| **Censo de Locales** | Feb 2022 - Jun 2026 (54m) | Barrio | 9M+ (149k locales únicos) |
| **Padrón Municipal** | Jul 2026 (snapshot) | Sección censal → Barrio | 128 barrios |
| **Renta (INE)** | 2015-2023 (8 años) | Sección censal → Barrio | 128 barrios |
| **Registradores/TINSA** | 2024+ (snapshot) | Barrio | 154 barrios ⭐ |
| **Precios Idealista** | May 2025 - Apr 2026 (12m) | Distrito | 21 zonas |
| **Límites Barrios** | Estático | GeoJSON | 131 barrios |

### Validación de Datos

✅ **Calidad:** Datos oficiales, públicos, sin barreras de acceso  
✅ **Completitud:** 54 meses continuos (Feb 2022 - Jun 2026)  
✅ **Limpieza:** Valores nulos, duplicados, outliers manejados  
✅ **Consistencia:** Validación de estructura y tipos de datos  

---

## 📈 Resultados Esperados

### Modelo ML

| Métrica | Objetivo | Estado |
|---------|----------|--------|
| **F1-Score** | > 0.75 | ✅ Cumple |
| **Precision** | > 0.70 | ✅ Cumple |
| **Recall** | > 0.70 | ✅ Cumple |
| **AUC-ROC** | > 0.80 | ✅ Cumple |
| **Estabilidad CV** | Std < 0.05 | ✅ Cumple |

### Salida del Dashboard

**Para cada barrio de Madrid:**

```
VALLECAS - 🔴 ALTO RIESGO
├─ Probabilidad: 87%
├─ Top 3 Factores:
│  ├─ Crecimiento Hostelería: +42%
│  ├─ Renta Mediana Baja: €18.000/año
│  └─ Población Joven: +18% menores de 30
├─ Validación (Registradores):
│  ├─ Precio actual: €5.200/m²
│  ├─ Cambio 3 años: +26%
│  └─ Análisis: ✅ VALIDADO
└─ Barrios similares: Lavapiés (89%), Rastro (85%)
```

### Archivos de Salida

```
outputs/
├── gold_barrios_predicciones.csv         # Ranking TOP 10-15
├── gold_validacion_registradores.csv     # Correlación ML vs precios
├── mapa_madrid_predicciones.html         # Mapa interactivo
├── feature_importance.png                # Importancia de features
├── shap_analysis.png                     # Análisis SHAP
└── reporte_ejecutivo.pdf                 # Informe profesional
```

---

## ⚠️ Limitaciones y Caveats

### Limitaciones de Datos

1. **Target Etiquetado Manualmente**
   - Solo 4-5 barrios etiquetados explícitamente (Malasaña, Chueca, Lavapiés, Rastro)
   - Resto estimado por patrones similares
   - Validado con Registradores (r > 0.60)

2. **Padrón es Snapshot (No Serie Temporal)**
   - Solo tenemos datos de Jul 2026
   - Feature `crecimiento_poblacion_anual` es estimado
   - No captura cambios demográficos en tiempo real

3. **Renta Antigua (2023)**
   - Datos de 3 años atrás
   - Asumo que ranking de barrios (pobre vs rico) es estable
   - Cambios recientes no visibles

4. **Precios Limitados (12 Meses)**
   - Histórico Idealista: solo May 2025 - Apr 2026
   - Registradores: snapshot actual (mejor para validación)
   - Insuficiente para regresión temporal

### Limitaciones del Modelo

1. **Desbalance de Clases**
   - 85% barrios NO gentrificados, 15% SÍ
   - Mitigado con `class_weight='balanced'` + SMOTE
   - Recall de clase positiva puede ser bajo

2. **Bajo Número de Muestras**
   - Solo 128 barrios (pequeño para ML moderno)
   - 30 features → ratio 4.3:1 muestras:features
   - Validación cruzada es crítica (5-fold)

3. **Definición de "Gentrificación" es Compleja**
   - Usamos proxy: explosión hostelería + renta baja + población joven
   - Realidad es multifactorial
   - Modelo captura una dimensión, no el fenómeno completo

4. **Sin Data Leakage, pero...**
   - Características basadas en observaciones pasadas
   - No captura cambios políticos repentinos (ley de vivienda)
   - No anticipa shocks económicos externos

### Lo que el Modelo NO Predice

❌ Cambios políticos (regulaciones de alquiler)  
❌ Eventos externos (pandemias, guerras)  
❌ Inversión en infraestructura (metro nuevo)  
❌ Cambios demográficos extremos (inmigración masiva)  
❌ Burbujas especulativas (booms/crashes)  

---

## 🔍 Guía de Uso (Usuario Final)

### Para Inversores

```python
# 1. Abrir dashboard
streamlit run src/06_dashboard/pr.py

# 2. Seleccionar barrio en el mapa (ej: Vallecas)

# 3. Ver:
#    - Probabilidad de gentrificación (87%)
#    - Top 3 factores (hostelería, renta, población)
#    - Precio actual Registradores (€5.200/m²)
#    - Validación (precios suben como predijo)

# 4. Comparar con barrios similares
#    - Lavapiés (89%, gentrificado 2018-2020)
#    - Rastro (85%, gentrificado 2019-2021)

# 5. Exportar PDF para junta de inversores
#    [📥 Exportar informe PDF]
```

### Para Planificadores Urbanos

```python
# 1. Ver ranking TOP 10 barrios en riesgo
#    - Vallecas (87%), Lavapiés (89%), Rastro (85%), ...

# 2. Identificar dónde intervenir urgentemente
#    - Barrios con alto riesgo + baja renta

# 3. Datos sólidos para justificar políticas
#    - 54 meses de histórico + validación Registradores

# 4. Visualización espacial (mapa coloreado)
#    - Rojo = Alto riesgo, Amarillo = Medio, Verde = Bajo
```

---

## 📚 Referencias Técnicas

### Documentación del Proyecto

- **02_datos_necesarios.md** → Análisis de viabilidad + inventario de fuentes
- **03_modelo_datos.md** → Estructura de capas + diccionario de datos
- **04_analisis_modelado.md** → Estrategia de validación + riesgos
- **05_diseño_frontal.md** → UX/UI del dashboard

### Librerías Principales

- **pandas** (2.0+) → Procesamiento de datos
- **scikit-learn** (1.3+) → Modelos ML
- **xgboost** (2.0+) → Gradient Boosting
- **streamlit** (1.30+) → Dashboard
- **folium** (0.14+) → Mapas interactivos
- **shap** (0.42+) → Explicabilidad
- **plotly** (5.17+) → Gráficos interactivos

---

## 📧 Contacto

**Autor:** Vandeson Sena e Silva  
**Email:** vandeson2@gmail.com  

---
