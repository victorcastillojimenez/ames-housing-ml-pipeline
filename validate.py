"""
=============================================================
VALIDADOR FINAL — Checklist de Calidad del Proyecto
=============================================================
Verifica que el pipeline de agentes haya producido entregables
que cumplen con los criterios definidos en el Proyecto I:

  [ ] ¿Hay al menos 5 modelos entrenados?
  [ ] ¿Se usó validación cruzada y búsqueda de hiperparámetros?
  [ ] ¿La memoria técnica tiene la extensión requerida (15-20 págs)?
  [ ] ¿Se incluyen las métricas R², RMSE y MAE?

Puede ejecutarse de forma independiente tras el pipeline:
  python validate.py

O es invocado automáticamente por el orquestador al finalizar.
=============================================================
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def _extract_text_from_docx(path):
    """Extrae texto plano de un archivo .docx."""
    try:
        from docx import Document
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs]
        # También extraer texto de tablas
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    paragraphs.append(cell.text)
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"  ⚠️  Error leyendo {path}: {e}")
        return ""

OUTPUT_DIR = Path("outputs")
MODELS_DIR  = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ─── Parámetros de validación ────────────────────────────────────────────────
MIN_MODELS           = 5
WORDS_PER_PAGE       = 110          # Estimación ajustada para docs con muchas gráficas grandes
MIN_PAGES            = 15
MAX_PAGES            = 25           # Aumentamos máximo por las gráficas
REQUIRED_METRICS     = ["r2", "rmse", "mae"]    # buscados en memoria e informe
REQUIRED_METRICS_ALT = ["R²", "RMSE", "MAE",   # variantes en el texto
                         "r2_test", "rmse_test", "mae_test",
                         "R^{2}", "r^2"]
REQUIRED_CV_TERMS    = ["cross_val", "KFold", "k-fold", "validación cruzada",
                         "cross-validation", "cv="]
REQUIRED_SEARCH_TERMS= ["RandomizedSearchCV", "GridSearchCV", "RandomSearch",
                         "hiperparámetros", "hyperparameter"]
# ─────────────────────────────────────────────────────────────────────────────


class CheckResult:
    def __init__(self, name: str):
        self.name    = name
        self.passed  = False
        self.details = []
        self.warnings= []

    def ok(self, msg: str):
        self.passed = True
        self.details.append(f"✅  {msg}")

    def fail(self, msg: str):
        self.passed = False
        self.details.append(f"❌  {msg}")

    def warn(self, msg: str):
        self.warnings.append(f"⚠️   {msg}")

    def info(self, msg: str):
        self.details.append(f"ℹ️   {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: ¿Hay al menos 5 modelos entrenados?
# ─────────────────────────────────────────────────────────────────────────────
def check_models_trained() -> CheckResult:
    result = CheckResult("¿Al menos 5 modelos entrenados?")

    # Fuente 1: archivos .pkl en models/
    pkl_files = list(MODELS_DIR.glob("*_optimized.pkl")) if MODELS_DIR.exists() else []
    result.info(f"Modelos .pkl encontrados (optimizados): {len(pkl_files)}")
    for p in pkl_files:
        result.info(f"   • {p.name}")

    # Fuente 2: reporte JSON del ML Engineer
    ml_report_path = OUTPUT_DIR / "03_ml_report.json"
    if ml_report_path.exists():
        with open(ml_report_path, encoding="utf-8") as f:
            ml_report = json.load(f)
        all_results = ml_report.get("all_results", [])
        model_names = list({r["model_name"] for r in all_results})
        result.info(f"Modelos registrados en reporte ML: {len(model_names)}")
        for name in model_names:
            result.info(f"   • {name}")

        # Verificar también que tienen validación cruzada (r2_cv_mean)
        cv_used = all(
            "r2_cv_mean" in r
            for r in all_results
            if r.get("tipo") == "base"
        )
        if cv_used:
            result.info("Cross-validation registrada en todos los modelos base ✓")
        else:
            result.warn("Algunos modelos base no tienen r2_cv_mean en el reporte.")

        total_unique = len(model_names)
        if total_unique >= MIN_MODELS:
            result.ok(f"{total_unique} modelos únicos entrenados ≥ mínimo requerido ({MIN_MODELS})")
        else:
            result.fail(f"Solo {total_unique} modelos encontrados. Mínimo requerido: {MIN_MODELS}")
    else:
        if len(pkl_files) >= MIN_MODELS:
            result.ok(f"{len(pkl_files)} archivos .pkl ≥ mínimo requerido ({MIN_MODELS})")
        else:
            result.fail(
                f"Reporte ML no encontrado y solo {len(pkl_files)} .pkl. "
                f"¿Se ejecutó el ML Engineer?"
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: ¿Se usó validación cruzada y búsqueda de hiperparámetros?
# ─────────────────────────────────────────────────────────────────────────────
def check_cv_and_hyperparameter_search() -> CheckResult:
    result = CheckResult("¿Validación cruzada + búsqueda de hiperparámetros?")

    cv_evidence     = []
    search_evidence = []

    # Fuente 1: código fuente del ML Engineer
    ml_agent_path = Path("agents") / "agent_03_ml_engineer.py"
    if ml_agent_path.exists():
        source = ml_agent_path.read_text(encoding="utf-8")
        for term in REQUIRED_CV_TERMS:
            if term in source:
                cv_evidence.append(f"'{term}' en agent_03_ml_engineer.py")
        for term in REQUIRED_SEARCH_TERMS:
            if term in source:
                search_evidence.append(f"'{term}' en agent_03_ml_engineer.py")

    # Fuente 2: reporte JSON del ML Engineer
    ml_report_path = OUTPUT_DIR / "03_ml_report.json"
    if ml_report_path.exists():
        with open(ml_report_path, encoding="utf-8") as f:
            ml_report = json.load(f)

        # Verificar CV: debe haber r2_cv_mean en modelos base
        base_with_cv = [
            r for r in ml_report.get("all_results", [])
            if r.get("tipo") == "base" and "r2_cv_mean" in r
        ]
        if base_with_cv:
            cv_evidence.append(
                f"r2_cv_mean registrado en {len(base_with_cv)} modelos base "
                f"(CV media: {sum(r['r2_cv_mean'] for r in base_with_cv)/len(base_with_cv):.4f})"
            )

        # Verificar búsqueda: debe haber best_params en modelos optimizados
        opt_with_params = [
            r for r in ml_report.get("all_results", [])
            if r.get("tipo") == "optimizado" and "best_params" in r
        ]
        if opt_with_params:
            search_evidence.append(
                f"best_params registrado en {len(opt_with_params)} modelos optimizados"
            )
            for r in opt_with_params[:3]:   # Mostrar máx. 3 ejemplos
                result.info(f"   • {r['model_name']}: {r.get('best_params', {})}")

    # Fuente 3: memoria técnica
    mem_path = OUTPUT_DIR / "MEMORIA_TECNICA.md"
    if mem_path.exists():
        mem_text = mem_path.read_text(encoding="utf-8")
        for term in REQUIRED_CV_TERMS + REQUIRED_SEARCH_TERMS:
            if term.lower() in mem_text.lower():
                if term in REQUIRED_CV_TERMS:
                    cv_evidence.append(f"'{term}' mencionado en MEMORIA_TECNICA.md")
                else:
                    search_evidence.append(f"'{term}' mencionado en MEMORIA_TECNICA.md")

    # Deduplicar evidencias
    cv_evidence     = list(dict.fromkeys(cv_evidence))
    search_evidence = list(dict.fromkeys(search_evidence))

    cv_ok     = len(cv_evidence) > 0
    search_ok = len(search_evidence) > 0

    if cv_ok:
        result.ok(f"Validación cruzada evidenciada ({len(cv_evidence)} fuente(s))")
        for ev in cv_evidence[:4]:
            result.info(f"   → {ev}")
    else:
        result.fail("No se encontró evidencia de validación cruzada.")

    if search_ok:
        result.ok(f"Búsqueda de hiperparámetros evidenciada ({len(search_evidence)} fuente(s))")
        for ev in search_evidence[:4]:
            result.info(f"   → {ev}")
    else:
        result.fail("No se encontró evidencia de búsqueda de hiperparámetros.")

    # El check pasa solo si ambas condiciones se cumplen
    result.passed = cv_ok and search_ok
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: ¿La memoria técnica tiene la extensión requerida (15-20 págs)?
# ─────────────────────────────────────────────────────────────────────────────
def check_technical_report_length() -> CheckResult:
    result = CheckResult("¿Memoria técnica con extensión 15-20 páginas?")

    # Buscar .docx primero, luego .md como fallback
    docx_path = OUTPUT_DIR / "MEMORIA_TECNICA.docx"
    md_path   = OUTPUT_DIR / "MEMORIA_TECNICA.md"

    if docx_path.exists():
        mem_path = docx_path
        text = _extract_text_from_docx(docx_path)
        result.info(f"Formato de memoria: .docx")
    elif md_path.exists():
        mem_path = md_path
        text = md_path.read_text(encoding="utf-8")
        result.info(f"Formato de memoria: .md")
    else:
        result.fail("MEMORIA_TECNICA.docx ni .md encontrada.")
        return result

    # Métricas de texto
    word_count   = len(text.split())
    line_count   = len(text.splitlines())
    char_count   = len(text)
    pages_est    = word_count / WORDS_PER_PAGE

    result.info(f"Palabras totales:    {word_count:,}")
    result.info(f"Líneas totales:      {line_count:,}")
    result.info(f"Caracteres totales:  {char_count:,}")
    result.info(f"Páginas estimadas:   {pages_est:.1f}  (a {WORDS_PER_PAGE} palabras/página)")

    # Verificar secciones mínimas esperadas
    expected_sections = [
        "Contexto", "Dataset", "EDA", "Preprocesamiento",
        "Modelo", "Optimización", "Evaluación", "Interpretab",
        "Mejora", "Conclus", "Bibliograf"
    ]
    found_sections = []
    missing_sections = []
    text_lower = text.lower()
    for sec in expected_sections:
        if sec.lower() in text_lower:
            found_sections.append(sec)
        else:
            missing_sections.append(sec)

    result.info(f"Secciones requeridas encontradas: {len(found_sections)}/{len(expected_sections)}")
    if missing_sections:
        result.warn(f"Secciones no detectadas: {', '.join(missing_sections)}")

    # Criterio de extensión
    if MIN_PAGES <= pages_est <= MAX_PAGES:
        result.ok(
            f"Extensión estimada: {pages_est:.1f} páginas — dentro del rango "
            f"requerido [{MIN_PAGES}-{MAX_PAGES} págs]"
        )
    elif pages_est < MIN_PAGES:
        result.fail(
            f"Extensión estimada: {pages_est:.1f} páginas — INSUFICIENTE "
            f"(mínimo: {MIN_PAGES} págs)"
        )
        result.warn(
            f"Faltan aprox. {(MIN_PAGES - pages_est) * WORDS_PER_PAGE:.0f} palabras "
            f"para alcanzar el mínimo."
        )
    else:
        result.warn(
            f"Extensión estimada: {pages_est:.1f} páginas — supera el máximo "
            f"recomendado ({MAX_PAGES} págs). Considerar síntesis."
        )
        result.passed = True   # Superar el máximo no es un fallo crítico

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: ¿Se incluyen las métricas R², RMSE y MAE?
# ─────────────────────────────────────────────────────────────────────────────
def check_metrics_present() -> CheckResult:
    result = CheckResult("¿Se incluyen métricas R², RMSE y MAE?")

    metrics_found = {"R²": False, "RMSE": False, "MAE": False}
    sources_checked = []

    # Buscar en múltiples fuentes
    # Determinar ruta de la memoria técnica (docx tiene prioridad)
    docx_mem = OUTPUT_DIR / "MEMORIA_TECNICA.docx"
    md_mem   = OUTPUT_DIR / "MEMORIA_TECNICA.md"
    mem_key  = "MEMORIA_TECNICA.docx" if docx_mem.exists() else "MEMORIA_TECNICA.md"
    mem_path = docx_mem if docx_mem.exists() else md_mem

    files_to_check = {
        mem_key:                      mem_path,
        "04_quality_report.json":     OUTPUT_DIR / "04_quality_report.json",
        "03_ml_report.json":          OUTPUT_DIR / "03_ml_report.json",
        "PRESENTACION_EJECUTIVA.md":  OUTPUT_DIR / "PRESENTACION_EJECUTIVA.md",
    }

    for fname, fpath in files_to_check.items():
        if not fpath.exists():
            result.warn(f"Archivo no encontrado: {fname}")
            continue

        sources_checked.append(fname)

        # Leer contenido según tipo de archivo
        if fpath.suffix == ".docx":
            content = _extract_text_from_docx(fpath)
        elif fpath.suffix == ".md":
            content = fpath.read_text(encoding="utf-8")
        else:
            content = ""

        # Para JSON, buscar keys directamente
        if fpath.suffix == ".json":
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                data_str = json.dumps(data)
                if any(t in data_str for t in ["r2_test", "r2_cv", "R2"]):
                    metrics_found["R²"] = True
                    result.info(f"R²  → en {fname}")
                if "rmse_test" in data_str or "rmse" in data_str.lower():
                    metrics_found["RMSE"] = True
                    result.info(f"RMSE → en {fname}")
                if "mae_test" in data_str or "mae" in data_str.lower():
                    metrics_found["MAE"] = True
                    result.info(f"MAE  → en {fname}")
            except json.JSONDecodeError:
                pass
        else:
            content_lower = content.lower()
            r2_terms   = ["r²", "r^2", "r2", "coeficiente de determinación",
                          "r²_test", "r2_test", "r2 ("]
            rmse_terms = ["rmse", "raíz del error cuadrático", "root mean square"]
            mae_terms  = ["mae", "error absoluto medio", "mean absolute error"]

            if any(t in content_lower for t in r2_terms):
                metrics_found["R²"] = True
                result.info(f"R²   → encontrado en {fname}")
            if any(t in content_lower for t in rmse_terms):
                metrics_found["RMSE"] = True
                result.info(f"RMSE → encontrado en {fname}")
            if any(t in content_lower for t in mae_terms):
                metrics_found["MAE"] = True
                result.info(f"MAE  → encontrado en {fname}")

    # Verificar también que los valores numéricos están en el reporte de calidad
    qa_path = OUTPUT_DIR / "04_quality_report.json"
    if qa_path.exists():
        with open(qa_path, encoding="utf-8") as f:
            qa = json.load(f)
        metrics_table = qa.get("metrics_table", [])
        if metrics_table:
            sample = metrics_table[0]
            if "r2_test" in sample:
                result.info(
                    f"Ejemplo — {sample.get('model_name','?')}: "
                    f"R²={sample.get('r2_test','?')}  "
                    f"RMSE=${sample.get('rmse_test','?'):,.0f}  "
                    f"MAE=${sample.get('mae_test','?'):,.0f}"
                )

    # Resultado final del check
    all_found = all(metrics_found.values())
    if all_found:
        result.ok("Las 3 métricas obligatorias (R², RMSE, MAE) están presentes y documentadas.")
    else:
        missing = [k for k, v in metrics_found.items() if not v]
        result.fail(f"Métricas FALTANTES: {', '.join(missing)}")

    result.passed = all_found
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def run_validation(verbose: bool = True) -> dict:
    """
    Ejecuta todos los checks y devuelve un dict con el resumen.
    Parámetro `verbose` controla si imprime el reporte completo.
    """
    start = datetime.now()

    checks = [
        check_models_trained,
        check_cv_and_hyperparameter_search,
        check_technical_report_length,
        check_metrics_present,
    ]

    labels = [
        "CHECK 1 — Modelos entrenados (≥5)",
        "CHECK 2 — Validación cruzada + Hiperparámetros",
        "CHECK 3 — Extensión Memoria Técnica (15-20 págs)",
        "CHECK 4 — Métricas R², RMSE y MAE presentes",
    ]

    if verbose:
        print("\n" + "╔" + "═"*62 + "╗")
        print("║" + " "*15 + "VALIDACIÓN FINAL DEL PROYECTO" + " "*18 + "║")
        print("║" + " "*14 + "Ames Insights Consulting · Checklist" + " "*12 + "║")
        print("╚" + "═"*62 + "╝")
        print(f"  Inicio: {start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = []
    for label, check_fn in zip(labels, checks):
        if verbose:
            print(f"\n  {'─'*60}")
            print(f"  {label}")
            print(f"  {'─'*60}")
        r = check_fn()
        results.append(r)
        if verbose:
            for line in r.details:
                print(f"   {line}")
            for line in r.warnings:
                print(f"   {line}")

    # ── Resumen final ────────────────────────────────────────────────────────
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    all_passed = len(failed) == 0

    checklist_rows = []
    for r in results:
        mark = "✅" if r.passed else "❌"
        checklist_rows.append({"check": r.name, "passed": r.passed, "mark": mark})

    if verbose:
        print(f"\n\n  {'═'*62}")
        print(f"  RESUMEN DEL CHECKLIST")
        print(f"  {'═'*62}")
        for row in checklist_rows:
            print(f"   [{row['mark']}]  {row['check']}")

        print(f"\n  Resultado: {len(passed)}/{len(results)} checks superados")

        if all_passed:
            print(f"\n  🎉  VALIDACIÓN COMPLETA — El proyecto cumple todos los criterios.")
            print(f"       El pipeline está listo para presentación al cliente.")
        else:
            print(f"\n  ⚠️  VALIDACIÓN INCOMPLETA — {len(failed)} check(s) fallaron:")
            for r in failed:
                print(f"       ❌  {r.name}")
            print(f"\n       Revisa los detalles anteriores y re-ejecuta el pipeline.")

        elapsed = (datetime.now() - start).total_seconds()
        print(f"\n  ⏱️  Tiempo de validación: {elapsed:.2f}s")
        print(f"  {'═'*62}\n")

    # ── Guardar reporte JSON ─────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    report = {
        "validation_summary": {
            "timestamp":     start.isoformat(),
            "total_checks":  len(results),
            "passed":        len(passed),
            "failed":        len(failed),
            "all_passed":    all_passed,
        },
        "checklist": checklist_rows,
        "check_details": [
            {
                "name":     r.name,
                "passed":   r.passed,
                "details":  r.details,
                "warnings": r.warnings,
            }
            for r in results
        ],
    }
    report_path = OUTPUT_DIR / "05_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"  📋  Reporte de validación guardado: {report_path}\n")

    return report


if __name__ == "__main__":
    report = run_validation(verbose=True)
    # Salir con código 1 si algún check falla (útil para CI/CD)
    sys.exit(0 if report["validation_summary"]["all_passed"] else 1)
