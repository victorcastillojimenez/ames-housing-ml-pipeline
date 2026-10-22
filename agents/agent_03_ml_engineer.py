"""
=============================================================
AGENTE 3: ML Engineer  |  Fases 4 & 5 - Modelado y Optimización
=============================================================
System Prompt:
  Eres el responsable de modelado. Debes entrenar al menos 5 modelos:
  Regresión Lineal, Árbol de Decisión, Random Forest, Gradient Boosting
  y XGBoost. Es obligatorio usar Validación Cruzada (k-fold) y optimizar
  hiperparámetros con GridSearchCV o RandomizedSearchCV. Compara el
  rendimiento del modelo base contra el optimizado y registra los tiempos
  de entrenamiento.
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")

AGENT_NAME  = "ML Engineer"
OUTPUT_DIR  = Path("outputs")
MODELS_DIR  = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"

KF = KFold(n_splits=5, shuffle=True, random_state=42)

# ──────────────────────────────────────────────────────────────────────────────

def _load_data():
    X_train = pd.read_csv(OUTPUT_DIR / "X_train.csv")
    X_test  = pd.read_csv(OUTPUT_DIR / "X_test.csv")
    X_train_sc = pd.read_csv(OUTPUT_DIR / "X_train_scaled.csv")
    X_test_sc  = pd.read_csv(OUTPUT_DIR / "X_test_scaled.csv")
    y_train = pd.read_csv(OUTPUT_DIR / "y_train.csv").squeeze()
    y_test  = pd.read_csv(OUTPUT_DIR / "y_test.csv").squeeze()
    print(f"  📂  Datos cargados: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test


def _train_and_evaluate(name, model, X_tr, X_te, y_tr, y_te, use_scaled=False) -> dict:
    """Entrena modelo base + CV y retorna métricas."""
    Xtr = X_tr
    Xte = X_te

    t0 = time.time()
    model.fit(Xtr, y_tr)
    train_time = time.time() - t0

    # Predicciones en escala log → revertir con expm1
    y_pred_train = model.predict(Xtr)
    y_pred_test  = model.predict(Xte)

    r2_train  = r2_score(y_tr, y_pred_train)
    r2_test   = r2_score(y_te, y_pred_test)

    # RMSE en escala original ($)
    rmse_test = np.sqrt(mean_squared_error(
        np.expm1(y_te), np.expm1(y_pred_test)))
    mae_test  = mean_absolute_error(
        np.expm1(y_te), np.expm1(y_pred_test))

    # Cross-validation R²
    cv_scores = cross_val_score(model, Xtr, y_tr, cv=KF, scoring="r2")

    return {
        "model_name":    name,
        "r2_train":      round(r2_train, 4),
        "r2_test":       round(r2_test, 4),
        "r2_cv_mean":    round(cv_scores.mean(), 4),
        "r2_cv_std":     round(cv_scores.std(), 4),
        "rmse_test":     round(rmse_test, 2),
        "mae_test":      round(mae_test, 2),
        "train_time_s":  round(train_time, 2),
        "overfitting":   round(r2_train - r2_test, 4),
    }


def _optimize_model(name, model, param_dist, X_tr, y_tr, n_iter=20) -> tuple:
    """RandomizedSearchCV para optimización de hiperparámetros."""
    print(f"    🔧  Optimizando {name} ({n_iter} iteraciones)...")
    t0 = time.time()
    search = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=KF,
        scoring="r2", n_jobs=-1, random_state=42, verbose=0
    )
    search.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    print(f"       Best R² CV: {search.best_score_:.4f}  |  Tiempo: {elapsed:.1f}s")
    print(f"       Mejores params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, elapsed


def _plot_comparison(results: list) -> None:
    """Barras horizontales: R² test de cada modelo."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(results).sort_values("r2_test", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_theme(style="darkgrid")

    # R² comparativa base vs optimizado
    base   = df_res[df_res["tipo"] == "base"]
    optim  = df_res[df_res["tipo"] == "optimizado"]

    x = np.arange(len(base))
    width = 0.35
    ax = axes[0]
    bars1 = ax.bar(x - width/2, base["r2_test"].values, width, label="Base", color="#4C72B0")
    bars2 = ax.bar(x + width/2, optim["r2_test"].values, width, label="Optimizado", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(base["model_name"].values, rotation=20, ha="right")
    ax.set_ylabel("R² (Test)")
    ax.set_title("R² Test: Modelo Base vs Optimizado")
    ax.legend()
    ax.set_ylim(0, 1)

    # RMSE comparativa
    ax2 = axes[1]
    ax2.bar(x - width/2, base["rmse_test"].values / 1000, width, label="Base", color="#C44E52")
    ax2.bar(x + width/2, optim["rmse_test"].values / 1000, width, label="Optimizado", color="#DD8452")
    ax2.set_xticks(x)
    ax2.set_xticklabels(base["model_name"].values, rotation=20, ha="right")
    ax2.set_ylabel("RMSE (miles $)")
    ax2.set_title("RMSE Test: Modelo Base vs Optimizado")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig05_model_comparison.png", dpi=150)
    plt.close()
    print("  📊  Figura 5 guardada: fig05_model_comparison.png")


def run(context: dict, reoptimize: bool = False) -> dict:
    print(f"\n{'='*60}")
    print(f"  ⚙️   AGENTE: {AGENT_NAME}")
    if reoptimize:
        print(f"  🔄  MODO: Re-optimización (solicitado por Quality Analyst)")
    print(f"  📅  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test = _load_data()

    # ── Definición de modelos base ──────────────────────────────────────────
    models_base = {
        "LinearRegression":    (LinearRegression(),        X_train_sc, X_test_sc),
        "DecisionTree":        (DecisionTreeRegressor(random_state=42), X_train, X_test),
        "RandomForest":        (RandomForestRegressor(n_estimators=50, random_state=42), X_train, X_test),
        "GradientBoosting":    (GradientBoostingRegressor(random_state=42), X_train, X_test),
        "XGBoost":             (xgb.XGBRegressor(random_state=42, verbosity=0), X_train, X_test),
        "StackingEnsemble":    (
            StackingRegressor(
                estimators=[
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                    ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0))
                ],
                final_estimator=Ridge()
            ), X_train, X_test
        ),
    }

    # ── Grids de hiperparámetros para RandomizedSearchCV ───────────────────
    param_grids = {
        "LinearRegression":  None,   # se usará Ridge con alpha
        "DecisionTree":      {"max_depth": [4, 6, 8, 12, None],
                              "min_samples_split": [2, 5, 10],
                              "min_samples_leaf":  [1, 2, 4]},
        "RandomForest":      {"n_estimators": [100, 200, 300],
                              "max_depth": [None, 10, 20],
                              "min_samples_split": [2, 5],
                              "max_features": ["sqrt", "log2"]},
        "GradientBoosting":  {"n_estimators": [100, 200],
                              "learning_rate": [0.05, 0.1, 0.15],
                              "max_depth": [3, 4, 5],
                              "subsample":    [0.7, 0.8, 1.0]},
        "XGBoost":           {"n_estimators": [100, 200, 300],
                              "learning_rate": [0.05, 0.1, 0.15],
                              "max_depth": [4, 6, 8],
                              "subsample":    [0.7, 0.8, 1.0],
                              "colsample_bytree": [0.6, 0.8, 1.0]},
        "StackingEnsemble":  None,
    }

    all_results = []

    # ── Fase 4: Entrenamiento de modelos BASE ───────────────────────────────
    print("\n  [Fase 4] Entrenando modelos BASE con validación cruzada 5-fold...\n")
    trained_base = {}
    for name, (model, Xtr, Xte) in models_base.items():
        print(f"    ▶  {name}")
        metrics = _train_and_evaluate(name, model, Xtr, Xte, y_train, y_test)
        metrics["tipo"] = "base"
        all_results.append(metrics)
        trained_base[name] = model
        joblib.dump(model, MODELS_DIR / f"{name}_base.pkl")
        print(f"       R²_train={metrics['r2_train']}  R²_test={metrics['r2_test']}  "
              f"CV={metrics['r2_cv_mean']}±{metrics['r2_cv_std']}  "
              f"RMSE=${metrics['rmse_test']:,.0f}  t={metrics['train_time_s']}s")

    # ── Fase 5: Optimización de hiperparámetros ─────────────────────────────
    print("\n  [Fase 5] Optimizando hiperparámetros (RandomizedSearchCV 5-fold)...\n")
    best_models = {}
    for name, (model, Xtr, Xte) in models_base.items():
        grid = param_grids[name]

        if name == "LinearRegression":
            # Usamos Ridge con búsqueda de alpha
            opt_model, best_params, opt_time = _optimize_model(
                "Ridge (LinearReg optimizado)",
                Ridge(),
                {"alpha": np.logspace(-3, 3, 20)},
                X_train_sc, y_train, n_iter=20
            )
            Xtr_opt, Xte_opt = X_train_sc, X_test_sc
        else:
            from sklearn.base import clone
            if grid is None:
                print(f"    🔧  Evaluando {name} (Sin grid adicional)...")
                t0 = time.time()
                opt_model = clone(model)
                opt_model.fit(Xtr, y_train)
                opt_time = time.time() - t0
                best_params = {"config": "Default Base Stacking"}
            else:
                opt_model, best_params, opt_time = _optimize_model(
                    name, clone(model),
                    grid, Xtr, y_train, n_iter=15
                )
            Xtr_opt, Xte_opt = Xtr, Xte

        metrics = _train_and_evaluate(
            name, opt_model, Xtr_opt, Xte_opt, y_train, y_test)
        metrics["tipo"] = "optimizado"
        metrics["best_params"] = best_params
        metrics["optimization_time_s"] = round(opt_time, 2)
        all_results.append(metrics)
        best_models[name] = opt_model
        joblib.dump(opt_model, MODELS_DIR / f"{name}_optimized.pkl")
        print(f"       R²_train={metrics['r2_train']}  R²_test={metrics['r2_test']}  "
              f"RMSE=${metrics['rmse_test']:,.0f}")

    # ── Guardar reporte de modelos ──────────────────────────────────────────
    _plot_comparison(all_results)

    report = {
        "all_results": all_results,
        "best_model_name": max(
            [r for r in all_results if r["tipo"] == "optimizado"],
            key=lambda x: x["r2_test"]
        )["model_name"],
        "timestamp": datetime.now().isoformat(),
    }
    report_path = OUTPUT_DIR / "03_ml_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  🏆  Mejor modelo: {report['best_model_name']}")
    print(f"  ✅  Reporte guardado en: {report_path}")
    print(f"  🔜  Enviando métricas al Quality & Performance Analyst...\n")

    context.update(report)
    context["agent_03_status"] = "OK"
    return context


if __name__ == "__main__":
    run({})
