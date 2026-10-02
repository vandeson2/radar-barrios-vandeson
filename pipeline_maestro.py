#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                       PIPELINE MAESTRO                                    ║
║                  Radar de Barrio - Flujo Completo                         ║
╚════════════════════════════════════════════════════════════════════════════╝

Ejecuta el flujo completo del proyecto en orden:
    1. Data Loading (54 meses raw)
    2. Data Cleaning (limpieza, normalización)
    3. Consolidación de Hostelería
    4. Mapeo de Barrios (fuzzy matching)
    5. Capa Gold (ingeniería de features)
    6. Generación de Feature Names
    7. Entrenamiento ML (Ensemble)
    8. Dashboard (Streamlit)

Uso:
    python pipeline_maestro.py --mode all        # Ejecutar todo
    python pipeline_maestro.py --mode hasta 5    # Hasta capa gold
    python pipeline_maestro.py --mode train-app  # Solo ML + dashboard
    python pipeline_maestro.py --mode dashboard  # Solo dashboard
    python pipeline_maestro.py --clean           # Limpiar caché
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
import json

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

PROYECTO_ROOT = Path(__file__).parent
LOGS_DIR = PROYECTO_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = LOGS_DIR / f'pipeline_maestro_{TIMESTAMP}.log'

# Orden de scripts
STAGES = {
    1: {
        'nombre': '01_data_loading',
        'descripcion': 'Cargar datos raw (54 meses)',
        'ruta': 'src/01_data/01_data_loading.py',
        'tipo': 'python'
    },
    2: {
        'nombre': '02_cleaning_main',
        'descripcion': 'Limpiar datos (valores nulos, duplicados)',
        'ruta': 'src/02_cleaning/03_cleaning_main.py',
        'tipo': 'python'
    },
    3: {
        'nombre': '03_consolidar_hosteleria',
        'descripcion': 'Consolidar 5 años de hostelería',
        'ruta': 'src/03_feature/04_consolidar_hosteleria.py',
        'tipo': 'python'
    },
    4: {
        'nombre': '04_mapeo_barrios',
        'descripcion': 'Mapear barrios (fuzzy matching)',
        'ruta': 'src/03_feature/mapeo/03_pipeline_mapeo_barrios.py',
        'tipo': 'python'
    },
    5: {
        'nombre': '05_capa_gold',
        'descripcion': 'Generar capa gold (128 barrios × 32 features)',
        'ruta': 'src/03_feature/_06_capa_gold.py',
        'tipo': 'python'
    },
    6:{
        'nombre': '06_enrichment',
        'description': 'Genera capa gold enriquecida',
        'ruta': 'src/04_enrichment/_05_pipeline_enriquecimiento.py',
        'tipo': 'python'
    },
    7: {
        'nombre': '06_feature_names',
        'descripcion': 'Generar lista de feature names',
        'ruta': 'src/05_ml_training/generar_feature_names.py',
        'tipo': 'python'
    },
    8: {
        'nombre': '07_train',
        'descripcion': 'Entrenar modelo Ensemble (SVM + RF + GB)',
        'ruta': 'src/05_ml_training/train.py',
        'tipo': 'python'
    },
    9: {
        'nombre': '08_dashboard',
        'descripcion': 'Ejecutar dashboard Streamlit',
        'ruta': 'src/06_dashboard/app.py',
        'tipo': 'streamlit'
    }
}

# ============================================================================
# LOGGING
# ============================================================================

def log_to_file(mensaje: str):
    """Log a archivo"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(mensaje + '\n')

def log_print(mensaje: str, nivel='INFO'):
    """Log a consola y archivo"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    msg_formateado = f"[{timestamp}] {nivel} - {mensaje}"
    print(msg_formateado)
    log_to_file(msg_formateado)

# ============================================================================
# EJECUCIÓN
# ============================================================================

class PipelineEjecutor:
    """Ejecutor del pipeline"""

    def __init__(self):
        self.resultados = {}
        self.inicio_total = datetime.now()

        log_print("=" * 80)
        log_print("PIPELINE MAESTRO - RADAR DE BARRIO")
        log_print("=" * 80)
        log_print(f"Raíz: {PROYECTO_ROOT}")
        log_print(f"Log: {LOG_FILE}")

    def ejecutar_script_python(self, stage_num: int) -> bool:
        """Ejecutar script Python"""
        stage = STAGES[stage_num]
        nombre = stage['nombre']
        ruta = PROYECTO_ROOT / stage['ruta']
        descripcion = stage['descripcion']

        log_print(f"\n{'='*80}")
        log_print(f"STAGE {stage_num}: {descripcion}")
        log_print(f"{'='*80}")
        log_print(f"Script: {ruta.name}")

        if not ruta.exists():
            log_print(f"Script no encontrado: {ruta}", 'ERROR')
            return False

        try:
            resultado = subprocess.run(
                [sys.executable, str(ruta)],
                cwd=PROYECTO_ROOT,
                capture_output=True,
                text=True,
                timeout=600
            )

            if resultado.returncode == 0:
                log_print(f"- {nombre} - COMPLETADO", 'SUCCESS')
                self.resultados[stage_num] = 'OK'
                return True
            else:
                log_print(f"- {nombre} - ERROR", 'ERROR')
                if resultado.stderr:
                    log_print(f"STDERR: {resultado.stderr[:300]}", 'ERROR')
                self.resultados[stage_num] = 'ERROR'
                return False

        except subprocess.TimeoutExpired:
            log_print(f"-{nombre} - TIMEOUT (>10min)", 'ERROR')
            self.resultados[stage_num] = 'TIMEOUT'
            return False
        except Exception as e:
            log_print(f"- {nombre} - EXCEPCIÓN: {str(e)[:200]}", 'ERROR')
            self.resultados[stage_num] = 'EXCEPCIÓN'
            return False

    def ejecutar_streamlit(self, stage_num: int):
        """Ejecutar Streamlit app"""
        stage = STAGES[stage_num]
        nombre = stage['nombre']
        ruta = PROYECTO_ROOT / stage['ruta']
        descripcion = stage['descripcion']

        log_print(f"\n{'='*80}")
        log_print(f"STAGE {stage_num}: {descripcion}")
        log_print(f"{'='*80}")
        log_print(f"App: {ruta.name}")
        log_print(f"URL: http://localhost:8501")
        log_print("(Presiona Ctrl+C para detener)")

        try:
            subprocess.run(
                ['streamlit', 'run', str(ruta)],
                cwd=PROYECTO_ROOT
            )
            self.resultados[stage_num] = 'OK'
        except KeyboardInterrupt:
            log_print("Dashboard detenido por usuario", 'INFO')
            self.resultados[stage_num] = 'OK (detenido por usuario)'
        except Exception as e:
            log_print(f"- {nombre} - Error: {str(e)[:200]}", 'ERROR')
            self.resultados[stage_num] = 'ERROR'

    def validar_requisitos(self) -> bool:
        """Validar que existan archivos necesarios"""
        log_print("\nValidando requisitos...")

        requisitos = {
            'config.py': 'Configuración',
            'src/': 'Código fuente',
            'data/': 'Directorio datos',
        }

        todos_ok = True
        for archivo, descripcion in requisitos.items():
            ruta = PROYECTO_ROOT / archivo
            if ruta.exists():
                log_print(f"- {archivo:20s} - {descripcion}")
            else:
                log_print(f"- {archivo:20s} - {descripcion}", 'WARNING')
                todos_ok = False

        return todos_ok

    def mostrar_resumen(self):
        """Mostrar resumen final"""
        log_print(f"\n{'='*80}")
        log_print("RESUMEN DE EJECUCIÓN")
        log_print(f"{'='*80}")

        for num in sorted(self.resultados.keys()):
            stage = STAGES[num]
            resultado = self.resultados[num]
            log_print(f"{stage['nombre']:30s} {resultado}")

        # Estadísticas
        ok_count = sum(1 for v in self.resultados.values() if '✅' in v)
        error_count = sum(1 for v in self.resultados.values() if '❌' in v)

        log_print(f"\n- Exitosos: {ok_count}")
        log_print(f"- Errores: {error_count}")

        duracion = datetime.now() - self.inicio_total
        minutos = duracion.total_seconds() / 60
        log_print(f"\n-  Duración total: {minutos:.1f} minutos")
        log_print(f"- Log: {LOG_FILE}")

        if error_count == 0 and ok_count > 0:
            log_print("\nPIPELINE COMPLETADO EXITOSAMENTE", 'SUCCESS')
        elif error_count > 0:
            log_print(f"\n-  {error_count} error(es) encontrado(s)", 'WARNING')

    def ejecutar_pipeline_completo(self):
        """Ejecutar pipeline completo"""
        log_print("\nModo: COMPLETO (todos los stages)")

        if not self.validar_requisitos():
            log_print("- Faltan requisitos", 'ERROR')
            return False

        # Ejecutar stages 1-7 (Python)
        for stage_num in range(1, 9):
            exito = self.ejecutar_script_python(stage_num)
            if not exito and stage_num <= 5:  # Stages críticos
                log_print(f"Stage crítico falló. Deteniendo...", 'WARNING')
                break

        # Stage 9 (Streamlit)
        self.ejecutar_streamlit(9)

        self.mostrar_resumen()

    def ejecutar_hasta_stage(self, stage_final: int):
        """Ejecutar hasta un stage específico"""
        log_print(f"\nModo: Ejecutar hasta stage {stage_final}")

        if not self.validar_requisitos():
            return False

        for stage_num in range(1, min(stage_final + 1, 10)):
            if stage_num == 9:  # No ejecutar dashboard
                log_print(f"\nStage {stage_num} es dashboard. Saltando...")
                continue

            exito = self.ejecutar_script_python(stage_num)
            if not exito and stage_num <= 5:
                log_print(f"Stage crítico falló. Deteniendo...", 'WARNING')
                break

        self.mostrar_resumen()

    def ejecutar_stages(self, stages_lista):
        """Ejecutar stages específicos"""
        log_print(f"\nModo: Custom - Stages {stages_lista}")

        for stage_num in stages_lista:
            if stage_num == 8:
                self.ejecutar_streamlit(stage_num)
            else:
                self.ejecutar_script_python(stage_num)

        self.mostrar_resumen()

    def limpiar_cache(self):
        """Limpiar caché"""
        import shutil

        log_print("Limpiando caché...")

        dirs_limpiar = [
            Path.home() / '.streamlit',
            PROYECTO_ROOT / '.pytest_cache',
            PROYECTO_ROOT / '__pycache__',
        ]

        for dir_path in dirs_limpiar:
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    log_print(f"Eliminado: {dir_path.name}")
                except:
                    pass

        log_print("Caché limpiado")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Punto de entrada"""
    parser = argparse.ArgumentParser(
        description='Pipeline Maestro - Radar de Barrio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python pipeline_maestro.py --mode all              # Todo
  python pipeline_maestro.py --mode hasta 5          # Hasta stage 5 (gold)
  python pipeline_maestro.py --mode train-app        # Stages 6-8
  python pipeline_maestro.py --mode dashboard        # Solo stage 8
  python pipeline_maestro.py --clean                 # Limpiar caché
        """
    )

    parser.add_argument(
        '--mode',
        choices=['all', 'hasta', 'train-app', 'dashboard', 'custom', 'gold'],
        default='all',
        help='Modo de ejecución'
    )

    parser.add_argument(
        '--stage',
        type=int,
        help='Para modo "hasta" o "custom", stage(s) a ejecutar'
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Limpiar caché y salir'
    )

    args = parser.parse_args()

    ejecutor = PipelineEjecutor()

    if args.clean:
        ejecutor.limpiar_cache()
        return 0

    # Modos de ejecución
    if args.mode == 'all':
        ejecutor.ejecutar_pipeline_completo()

    elif args.mode == 'hasta':
        if not args.stage:
            print("Error: --mode hasta requiere --stage N")
            return 1
        ejecutor.ejecutar_hasta_stage(args.stage)

    elif args.mode == 'gold':
        ejecutor.ejecutar_stages([4, 5, 6, 7, 8, 9])

    elif args.mode == 'train-app':
        ejecutor.ejecutar_stages([6, 7, 8,9])

    elif args.mode == 'dashboard':
        ejecutor.ejecutar_streamlit(9)

    return 0

if __name__ == '__main__':
    sys.exit(main())
