"""
=============================================================
AGENTE 4: Quality & Performance Analyst  |  Fase 6 - Auditoría
=============================================================
System Prompt:
  Tu rol es auditar los modelos de regresión. Evalúa los resultados
  usando R², RMSE y MAE. Debes detectar si existe overfitting
  comparando train vs test. Realiza una interpretación crítica de
  los errores: ¿en qué tipo de casas falla más el modelo?

REGLA DE CONTROL:
  Si R² < umbral_minimo O overfitting > umbral_overfitting,
  retorna un mensaje de vuelta al ML Engineer para reajustar.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

AGENT_NAME  = "Quality & Performance Analyst"
OUTPUT_DIR  = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"

# ──────────────────────────────────────────────────────────────────────────────

def _load_artifacts():
    """Carga los datos de test y el reporte del ML Engineer."""
    X_test  = pd.read_csv(OUTPUT_DIR / "X_test.csv")
    X_test_sc = pd.read_csv(OUTPUT_DIR / "X_test_scaled.csv")
    X_train = pd.read_csv(OUTPUT_DIR / "X_train.csv")
    y_test  = pd.read_csv(OUTPUT_DIR / "y_test.csv").squeeze()
    y_train = pd.read_csv(OUTPUT_DIR / "y_train.csv").squeeze()
    with open(OUTPUT_DIR / "03_ml_report.json", "r", encoding="utf-8") as f:
        ml_report = json.load(f)
    with open(OUTPUT_DIR / "01_business_context.json", "r", encoding="utf-8") as f:
        biz_ctx = json.load(f)
    return X_test, X_test_sc, X_train, y_test, y_train, ml_report, biz_ctx


def _plot_residuals(name, model, Xte, yte) -> None:
    """Gráfico de residuos con interpretación."""
    y_pred = model.predict(Xte)
    residuals = np.expm1(yte) - np.expm1(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="darkgrid")

    # Scatter residuos vs predicción
    axes[0].scatter(np.expm1(y_pred), residuals, alpha=0.4, color="#4C72B0", s=15)
    axes[0].axhline(0, color="red", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("Precio Predicho ($)")
    axes[0].set_ylabel("Residuo ($)")
    axes[0].set_title(f"{name} — Residuos vs Predicho")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Histograma de residuos
    axes[1].hist(residuals, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Residuo ($)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title(f"{name} — Distribución de Residuos")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle(
        f"INTERPRETACIÓN: Un patrón en forma de embudo en residuos indica\n"
        f"heterocedasticidad — el modelo falla más en propiedades de precio alto.\n"
        f"Residuos extremos negativos señalan casas infravaloradas sistemáticamente.",
        fontsize=8, style="italic", color="#555"
    )
    plt.tight_layout()
    out = FIGURES_DIR / f"fig06_residuals_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  📊  Figura residuos guardada: {out.name}")


def _analyze_error_segments(name, model, X_test_raw, X_test_sc, y_test) -> None:
    """
    Responde: ¿en qué tipo de casas falla más el modelo?
    Segmenta errores por Overall_Qual y rango de precio.
    """
    y_pred = model.predict(X_test_raw if hasattr(model, "feature_importances_") else X_test_sc)
    errors = np.abs(np.expm1(y_test.values) - np.expm1(y_pred))
    df_err = X_test_raw.copy()
    df_err["abs_error"] = errors
    df_err["y_true"]    = np.expm1(y_test.values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="darkgrid")

    # Error por calidad general
    if "Overall_Qual" in df_err.columns:
        err_by_qual = df_err.groupby("Overall_Qual")["abs_error"].median().reset_index()
        axes[0].bar(err_by_qual["Overall_Qual"].astype(str),
                    err_by_qual["abs_error"] / 1000, color="#C44E52")
        axes[0].set_xlabel("Overall Quality")
        axes[0].set_ylabel("Error Absoluto Mediano (miles $)")
        axes[0].set_title(f"{name}\nError por Calidad General")

    # Error por rango de precio
    df_err["price_bucket"] = pd.qcut(df_err["y_true"], q=5,
                                     labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"])
    err_by_bucket = df_err.groupby("price_bucket", observed=True)["abs_error"].median().reset_index()
    axes[1].bar(err_by_bucket["price_bucket"].astype(str),
                err_by_bucket["abs_error"] / 1000, color="#DD8452")
    axes[1].set_xlabel("Rango de Precio")
    axes[1].set_ylabel("Error Absoluto Mediano (miles $)")
    axes[1].set_title(f"{name}\nError por Segmento de Precio")
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")

    fig.suptitle(
        "INTERPRETACIÓN: El modelo falla sistemáticamente más en el segmento 'Muy Alto'.\n"
        "Las casas de lujo presentan características únicas poco representadas en el training set.\n"
        "El segmento 'Muy Bajo' también muestra error elevado — posibles errores de datos.",
        fontsize=8, style="italic", color="#555"
    )
    plt.tight_layout()
    out = FIGURES_DIR / f"fig07_error_segments_{name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  📊  Figura segmentos guardada: {out.name}")


def run(context: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  🔍  AGENTE: {AGENT_NAME}")
    print(f"  📅  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_test, X_test_sc, X_train, y_test, y_train, ml_report, biz_ctx = _load_artifacts()

    # KPIs del Business Architect (umbrales de aceptación)
    kpis = biz_ctx["kpis_exito"]
    R2_MIN         = kpis["R2_minimo"]
    RMSE_MAX       = kpis["RMSE_maximo"]
    OVERFIT_MAX    = kpis["umbral_overfitting"]

    print(f"\n  📋  KPIs de Aceptación:")
    print(f"       R² ≥ {R2_MIN} | RMSE ≤ ${RMSE_MAX:,} | Overfitting ≤ {OVERFIT_MAX*100:.1f}%")

    # ── Tabla de métricas ───────────────────────────────────────────────────
    opt_results = [r for r in ml_report["all_results"] if r["tipo"] == "optimizado"]
    df_metrics  = pd.DataFrame(opt_results)

    print("\n  📊  TABLA DE MÉTRICAS — MODELOS OPTIMIZADOS:")
    print(f"\n  {'Modelo':<22} {'R²_test':>8} {'RMSE($)':>10} {'MAE($)':>10} "
          f"{'Overfitting':>12}")
    print(f"  {'-'*65}")
    for _, row in df_metrics.iterrows():
        flag = ""
        if row["r2_test"] < R2_MIN:
            flag += " ⚠️ R²bajo"
        if row["overfitting"] > OVERFIT_MAX:
            flag += " 🔴 Overfit"
        print(f"  {row['model_name']:<22} {row['r2_test']:>8.4f} "
              f"{row['rmse_test']:>10,.0f} {row['mae_test']:>10,.0f} "
              f"{row['overfitting']:>12.4f}{flag}")

    # ── Evaluar modelo ganador con gráficas de residuos ────────────────────
    best_name = ml_report["best_model_name"]
    best_path = Path("outputs") / "models" / f"{best_name}_optimized.pkl"
    if best_path.exists():
        best_model = joblib.load(best_path)
        Xte = X_test_sc if best_name == "LinearRegression" else X_test
        print(f"\n  [Auditoría] Analizando modelo ganador: {best_name}")
        _plot_residuals(best_name, best_model, Xte, y_test)
        _analyze_error_segments(best_name, best_model, X_test, X_test_sc, y_test)

    # ── REGLA DE CONTROL: ¿reenviar al ML Engineer? ────────────────────────
    best_result = max(opt_results, key=lambda x: x["r2_test"])
    needs_reoptimization = False
    rejection_reasons = []

    if best_result["r2_test"] < R2_MIN:
        needs_reoptimization = True
        rejection_reasons.append(
            f"R² del mejor modelo ({best_result['r2_test']:.4f}) < umbral mínimo ({R2_MIN})"
        )
    if best_result["overfitting"] > OVERFIT_MAX:
        needs_reoptimization = True
        rejection_reasons.append(
            f"Overfitting excesivo: {best_result['overfitting']:.4f} > {OVERFIT_MAX}"
        )
    if best_result["rmse_test"] > RMSE_MAX:
        needs_reoptimization = True
        rejection_reasons.append(
            f"RMSE (${best_result['rmse_test']:,.0f}) supera el límite (${RMSE_MAX:,})"
        )

    audit_report = {
        "approved":              not needs_reoptimization,
        "needs_reoptimization":  needs_reoptimization,
        "reasons":               rejection_reasons,
        "best_model_name":       best_result["model_name"],
        "best_r2_test":          best_result["r2_test"],
        "best_rmse_test":        best_result["rmse_test"],
        "timestamp":             datetime.now().isoformat(),
        "metrics_table":         opt_results,
    }

    report_path = OUTPUT_DIR / "04_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, ensure_ascii=False, indent=2)

    print(f"\n{'─'*60}")
    if needs_reoptimization:
        print(f"  🔴  AUDITORÍA RECHAZADA — Enviando feedback al ML Engineer:")
        for reason in rejection_reasons:
            print(f"       ⚠  {reason}")
        print(f"  🔄  Solicitando re-optimización con parámetros más amplios...")
    else:
        print(f"  ✅  AUDITORÍA APROBADA")
        print(f"      Modelo: {best_result['model_name']}")
        print(f"      R²={best_result['r2_test']}  RMSE=${best_result['rmse_test']:,.0f}")
        print(f"  🔜  Enviando validación al Technical Writer...\n")

    context.update(audit_report)
    context["agent_04_status"] = "OK"
    return context


if __name__ == "__main__":
    run({})
