"""
Script independiente: Análisis de Error por Segmentos
Genera: outputs/figures/fig07_error_segments_<modelo>.png
Requiere: outputs/models/<modelo>_optimized.pkl, outputs/X_test.csv, outputs/y_test.csv
"""
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open("outputs/04_quality_report.json", "r", encoding="utf-8") as f:
    qa = json.load(f)
best_name = qa["best_model_name"]

model = joblib.load(f"outputs/models/{best_name}_optimized.pkl")
X_test = pd.read_csv("outputs/X_test.csv")
X_test_sc = pd.read_csv("outputs/X_test_scaled.csv")
y_test = pd.read_csv("outputs/y_test.csv").squeeze()

Xte = X_test_sc if best_name == "LinearRegression" else X_test
y_pred = model.predict(Xte)
errors = np.abs(np.expm1(y_test.values) - np.expm1(y_pred))

df_err = X_test.copy()
df_err["abs_error"] = errors
df_err["y_true"] = np.expm1(y_test.values)

sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Error por calidad general
if "Overall_Qual" in df_err.columns:
    err_by_qual = df_err.groupby("Overall_Qual")["abs_error"].median().reset_index()
    axes[0].bar(err_by_qual["Overall_Qual"].astype(str),
                err_by_qual["abs_error"] / 1000, color="#357083")
    axes[0].set_xlabel("Overall Quality")
    axes[0].set_ylabel("Error Absoluto Mediano (miles $)")
    axes[0].set_title(f"{best_name}\nError por Calidad General")

# Error por rango de precio
df_err["price_bucket"] = pd.qcut(df_err["y_true"], q=5,
                                 labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"])
err_by_bucket = df_err.groupby("price_bucket", observed=True)["abs_error"].median().reset_index()
axes[1].bar(err_by_bucket["price_bucket"].astype(str),
            err_by_bucket["abs_error"] / 1000, color="#9AA5AE")
axes[1].set_xlabel("Rango de Precio")
axes[1].set_ylabel("Error Absoluto Mediano (miles $)")
axes[1].set_title(f"{best_name}\nError por Segmento de Precio")
plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")

fig.suptitle(
    "INTERPRETACIÓN: El modelo falla sistemáticamente más en el segmento 'Muy Alto'.\n"
    "Las casas de lujo presentan características únicas poco representadas en el training set.\n"
    "El segmento 'Muy Bajo' también muestra error elevado — posibles errores de datos.",
    fontsize=8, style="italic", color="#9AA5AE"
)
plt.tight_layout()
safe_name = best_name.replace(" ", "_")
plt.savefig(FIGURES_DIR / f"fig07_error_segments_{safe_name}.png", dpi=150)
plt.close()
print(f"Guardada: fig07_error_segments_{safe_name}.png")
