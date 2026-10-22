"""
Script independiente: Top 20 Feature Importance del mejor modelo
Genera: outputs/figures/fig08_feature_importance.png
Requiere: outputs/models/<modelo>_optimized.pkl, outputs/02_eda_report.json
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

with open("outputs/02_eda_report.json", "r", encoding="utf-8") as f:
    eda = json.load(f)
feature_names = eda["feature_names"]

model = joblib.load(f"outputs/models/{best_name}_optimized.pkl")

if not hasattr(model, "feature_importances_"):
    print(f"El modelo {best_name} no tiene feature_importances_. Saliendo.")
    exit()

importances = model.feature_importances_
fi_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(20)
)

sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=fi_df, x="importance", y="feature", palette=["#357083", "#007B92", "#3DB9D3"], ax=ax)
ax.set_title(f"Top 20 Variables más Importantes — {best_name}", fontsize=13)
ax.set_xlabel("Importancia (Ganancia Media)")
fig.suptitle(
    "INTERPRETACIÓN PARA EL CLIENTE: Las variables estructurales (superficie, calidad general)\n"
    "dominan la predicción. Variables de ubicación y antigüedad tienen peso secundario\n"
    "pero son clave para las propiedades premium (riesgo identificado en Fase 1).",
    fontsize=8, style="italic", color="#9AA5AE"
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig08_feature_importance.png", dpi=150)
plt.close()
print(f"Guardada: {FIGURES_DIR / 'fig08_feature_importance.png'}")
