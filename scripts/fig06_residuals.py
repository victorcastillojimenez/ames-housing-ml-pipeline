"""
Script independiente: Gráfico de Residuos del mejor modelo
Genera: outputs/figures/fig06_residuals_<modelo>.png
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

# Leer el mejor modelo del reporte
with open("outputs/04_quality_report.json", "r", encoding="utf-8") as f:
    qa = json.load(f)
best_name = qa["best_model_name"]

# Cargar modelo y datos
model = joblib.load(f"outputs/models/{best_name}_optimized.pkl")
X_test = pd.read_csv("outputs/X_test.csv")
X_test_sc = pd.read_csv("outputs/X_test_scaled.csv")
y_test = pd.read_csv("outputs/y_test.csv").squeeze()

Xte = X_test_sc if best_name == "LinearRegression" else X_test
y_pred = model.predict(Xte)
residuals = np.expm1(y_test) - np.expm1(y_pred)

sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter residuos vs predicción
axes[0].scatter(np.expm1(y_pred), residuals, alpha=0.4, color="#007B92", s=15)
axes[0].axhline(0, color="#3DB9D3", linewidth=1.2, linestyle="--")
axes[0].set_xlabel("Precio Predicho ($)")
axes[0].set_ylabel("Residuo ($)")
axes[0].set_title(f"{best_name} — Residuos vs Predicho")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Histograma de residuos
axes[1].hist(residuals, bins=50, color="#3DB9D3", edgecolor="white", alpha=0.85)
axes[1].set_xlabel("Residuo ($)")
axes[1].set_ylabel("Frecuencia")
axes[1].set_title(f"{best_name} — Distribución de Residuos")
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

fig.suptitle(
    "INTERPRETACIÓN: Un patrón en forma de embudo en residuos indica\n"
    "heterocedasticidad — el modelo falla más en propiedades de precio alto.\n"
    "Residuos extremos negativos señalan casas infravaloradas sistemáticamente.",
    fontsize=8, style="italic", color="#9AA5AE"
)
plt.tight_layout()
safe_name = best_name.replace(" ", "_")
plt.savefig(FIGURES_DIR / f"fig06_residuals_{safe_name}.png", dpi=150)
plt.close()
print(f"Guardada: fig06_residuals_{safe_name}.png")
