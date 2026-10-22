"""
=============================================================
ORQUESTADOR PRINCIPAL — Proyecto Ames Housing
=============================================================
Coordina el flujo de mensajes entre los 5 agentes siguiendo
la metodología obligatoria del proyecto:

  1. Business Architect   → Contexto y Objetivos
  2. Data Scientist       → EDA, Limpieza y Preprocesamiento
  3. ML Engineer          → Entrenamiento + Optimización
  4. Quality Analyst      → Auditoría (con bucle de control)
     └─ Si R²<min o Overfit>max → Re-envía al ML Engineer
  5. Technical Writer     → Memoria técnica + Presentación

Cada agente recibe el 'context' acumulado del anterior
y lo enriquece con sus propios resultados.
=============================================================
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Añadir la carpeta agents al path
sys.path.insert(0, str(Path(__file__).parent / "agents"))

import agent_01_business_architect  as agent01
import agent_02_data_scientist      as agent02
import agent_03_ml_engineer         as agent03
import agent_04_quality_analyst     as agent04
import agent_05_technical_writer    as agent05
import validate                     as validator   # ← Validación final integrada

# ──────────────────────────────────────────────────────────────────────────────
MAX_REOPTIMIZATION_CYCLES = 2   # Máximo de veces que se puede reenviar al ML Engineer
LOG_FILE = Path("outputs") / "orchestrator_log.json"
# ──────────────────────────────────────────────────────────────────────────────


def _log_event(log: list, event: str, agent: str, status: str, elapsed: float = 0.0):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "event": event,
        "status": status,
        "elapsed_s": round(elapsed, 2),
    }
    log.append(entry)
    print(f"\n  📋  [{datetime.now().strftime('%H:%M:%S')}] "
          f"[{agent}] {event} → {status}")


def _print_banner():
    print("\n" + "╔" + "═"*60 + "╗")
    print("║" + " "*15 + "ORQUESTADOR DE AGENTES" + " "*23 + "║")
    print("║" + " "*10 + "Proyecto Ames Housing — Price Prediction" + " "*10 + "║")
    print("║" + " "*18 + "Ames Insights Consulting" + " "*18 + "║")
    print("╚" + "═"*60 + "╝")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Máx. ciclos de re-optimización: {MAX_REOPTIMIZATION_CYCLES}\n")


def run_pipeline():
    _print_banner()
    Path("outputs").mkdir(exist_ok=True)

    context = {}          # Contexto compartido entre agentes
    audit_log = []        # Log auditadle de la orquestación
    global_start = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # PASO 1: Business Architect — Contexto y Objetivos
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        context = agent01.run(context)
        _log_event(audit_log, "Contexto de negocio generado",
                   "Business Architect", "✅ OK", time.time() - t0)
    except Exception as e:
        _log_event(audit_log, f"ERROR: {e}", "Business Architect", "❌ FAIL")
        print(f"\n  ❌  PIPELINE ABORTADO en Business Architect: {e}")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # PASO 2: Data Scientist — EDA + Preprocesamiento
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        context = agent02.run(context)
        _log_event(audit_log, "EDA y preprocesamiento completados",
                   "Data Scientist", "✅ OK", time.time() - t0)
    except Exception as e:
        _log_event(audit_log, f"ERROR: {e}", "Data Scientist", "❌ FAIL")
        print(f"\n  ❌  PIPELINE ABORTADO en Data Scientist: {e}")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # PASO 3 + 4 (con bucle de control): ML Engineer ↔ Quality Analyst
    # ──────────────────────────────────────────────────────────────────────────
    approved = False
    cycle = 0

    while not approved and cycle < MAX_REOPTIMIZATION_CYCLES:
        cycle += 1
        cycle_label = f"Ciclo {cycle}/{MAX_REOPTIMIZATION_CYCLES}"

        # PASO 3: ML Engineer — Entrenamiento y Optimización
        print(f"\n  🔄  {cycle_label} — Iniciando ML Engineer...")
        t0 = time.time()
        try:
            context = agent03.run(context, reoptimize=(cycle > 1))
            _log_event(audit_log, f"Entrenamiento completado ({cycle_label})",
                       "ML Engineer", "✅ OK", time.time() - t0)
        except Exception as e:
            _log_event(audit_log, f"ERROR: {e}", "ML Engineer", "❌ FAIL")
            print(f"\n  ❌  PIPELINE ABORTADO en ML Engineer: {e}")
            return

        # PASO 4: Quality & Performance Analyst — Auditoría
        print(f"\n  🔍  {cycle_label} — Iniciando Quality Analyst...")
        t0 = time.time()
        try:
            context = agent04.run(context)
            _log_event(audit_log, f"Auditoría completada ({cycle_label})",
                       "Quality & Performance Analyst", "✅ OK", time.time() - t0)
        except Exception as e:
            _log_event(audit_log, f"ERROR: {e}", "Quality & Performance Analyst", "❌ FAIL")
            print(f"\n  ❌  PIPELINE ABORTADO en Quality Analyst: {e}")
            return

        approved = context.get("approved", False)

        if not approved:
            reasons = context.get("reasons", [])
            print(f"\n  🔄  BUCLE DE CONTROL ACTIVADO ({cycle_label}):")
            for r in reasons:
                print(f"       ⚠  {r}")
            if cycle < MAX_REOPTIMIZATION_CYCLES:
                print(f"  🔙  Re-enviando al ML Engineer para ajuste de hiperparámetros...")
            else:
                print(f"\n  ⚠️  Máximo de ciclos alcanzado ({MAX_REOPTIMIZATION_CYCLES}).")
                print(f"  ➡  Continuando con el mejor modelo disponible...")
                approved = True   # Forzar continuación al mejor resultado obtenido

    # ──────────────────────────────────────────────────────────────────────────
    # PASO 5: Technical Writer — Entregables finales
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        context = agent05.run(context)
        _log_event(audit_log, "Entregables finales generados",
                   "Technical Writer", "✅ OK", time.time() - t0)
    except Exception as e:
        _log_event(audit_log, f"ERROR: {e}", "Technical Writer", "❌ FAIL")
        print(f"\n  ❌  Error en Technical Writer: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # PASO 6: Validación Final — Checklist de Calidad del Proyecto
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "╔" + "═"*60 + "╗")
    print("║" + " "*14 + "VALIDACIÓN FINAL — CHECKLIST" + " "*18 + "║")
    print("╚" + "═"*60 + "╝")

    t0 = time.time()
    try:
        validation_report = validator.run_validation(verbose=True)
        val_elapsed = time.time() - t0
        val_passed  = validation_report["validation_summary"]["all_passed"]
        val_n_ok    = validation_report["validation_summary"]["passed"]
        val_total   = validation_report["validation_summary"]["total_checks"]
        val_status  = "✅ OK" if val_passed else f"⚠️ {val_total - val_n_ok} CHECKS FALLARON"

        _log_event(audit_log, f"Checklist de calidad: {val_n_ok}/{val_total} superados",
                   "Validator", val_status, val_elapsed)

        context["validation_passed"] = val_passed
        context["validation_checks"] = f"{val_n_ok}/{val_total}"

    except Exception as e:
        _log_event(audit_log, f"ERROR en validación: {e}", "Validator", "❌ FAIL")
        print(f"\n  ❌  Error en el validador: {e}")
        val_passed = False
        validation_report = {}

    # ──────────────────────────────────────────────────────────────────────────
    # RESUMEN FINAL
    # ──────────────────────────────────────────────────────────────────────────
    total_time = time.time() - global_start

    print("\n" + "╔" + "═"*60 + "╗")
    print("║" + " "*18 + "RESUMEN FINAL DEL PIPELINE" + " "*16 + "║")
    print("╚" + "═"*60 + "╝")

    print(f"\n  ⏱️  Tiempo total: {total_time/60:.1f} minutos ({total_time:.0f}s)")
    print(f"  🔄  Ciclos de re-optimización usados: {cycle - 1}")
    print(f"\n  ─── Resultados de los Agentes ───────────────────────────")
    print(f"  🏛️   Business Architect  : {context.get('agent_01_status', 'N/A')}")
    print(f"  🔬  Data Scientist       : {context.get('agent_02_status', 'N/A')}")
    print(f"  ⚙️   ML Engineer          : {context.get('agent_03_status', 'N/A')}")
    print(f"  🔍  Quality Analyst      : {context.get('agent_04_status', 'N/A')}")
    print(f"  📝  Technical Writer     : {context.get('agent_05_status', 'N/A')}")
    print(f"  ✅  Auditoría aprobada   : {context.get('approved', False)}")
    print(f"\n  ─── Métricas del Mejor Modelo ───────────────────────────")
    print(f"  🏆  Modelo ganador : {context.get('best_model_name', 'N/A')}")
    print(f"  📊  R²  (test)     : {context.get('best_r2_test', 0):.4f}")
    print(f"  💰  RMSE (test)    : ${context.get('best_rmse_test', 0):,.0f}")
    print(f"\n  ─── Checklist de Validación Final ───────────────────────")

    # Imprimir checklist con estado
    if validation_report:
        for item in validation_report.get("checklist", []):
            mark = "✅" if item["passed"] else "❌"
            print(f"  [{mark}]  {item['check']}")
    val_label = "✅ PROYECTO VALIDADO" if val_passed else "⚠️  VALIDACIÓN INCOMPLETA"
    print(f"\n  {val_label}  ({context.get('validation_checks', '?')} checks superados)")

    print(f"\n  ─── Entregables Generados ───────────────────────────────")
    deliverables = context.get("deliverables", {})
    label_map = {
        "memoria_tecnica": "📄 Memoria Técnica",
        "presentacion":    "🎤 Presentación",
        "figures_dir":     "📊 Gráficas",
    }
    for key, path in deliverables.items():
        print(f"  {label_map.get(key, key):<25}: {path}")

    print(f"\n  📋  Logs de auditoría    : {LOG_FILE}")
    print(f"  📋  Reporte validación   : outputs/05_validation_report.json")
    print()

    # ── Guardar log consolidado de auditoría ──────────────────────────────
    Path("outputs").mkdir(exist_ok=True)
    final_log = {
        "pipeline_summary": {
            "total_time_s":            round(total_time, 2),
            "reoptimization_cycles":   cycle - 1,
            "best_model":              context.get("best_model_name", "N/A"),
            "final_r2":                context.get("best_r2_test", 0),
            "final_rmse":              context.get("best_rmse_test", 0),
            "model_approved":          context.get("approved", False),
            "validation_passed":       val_passed,
            "validation_checks":       context.get("validation_checks", "?"),
            "timestamp":               datetime.now().isoformat(),
        },
        "agent_events":     audit_log,
        "deliverables":     deliverables,
        "validation_report": validation_report.get("validation_summary", {}),
    }
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(final_log, f, ensure_ascii=False, indent=2)
    print(f"  ✅  Log consolidado guardado en: {LOG_FILE}\n")

    # Código de salida útil para CI/CD
    return val_passed


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
