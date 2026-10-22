"""
Script independiente: Comparativa Base vs Optimizado (R² y RMSE)
Genera: outputs/figures/fig05_model_comparison.png
Requiere: outputs/03_ml_report.json (generado por el ML Engineer)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open("outputs/03_ml_report.json", "r", encoding="utf-8") as f:
    ml_report = json.load(f)

results = ml_report["all_results"]
df_res = pd.DataFrame(results)
base  = df_res[df_res["tipo"] == "base"].sort_values("model_name").reset_index(drop=True)
optim = df_res[df_res["tipo"] == "optimizado"].sort_values("model_name").reset_index(drop=True)

sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(base))
width = 0.35

# R² comparativa
ax = axes[0]
ax.bar(x - width/2, base["r2_test"].values, width, label="Base", color="#007B92")
ax.bar(x + width/2, optim["r2_test"].values, width, label="Optimizado", color="#3DB9D3")
ax.set_xticks(x)
ax.set_xticklabels(base["model_name"].values, rotation=20, ha="right")
ax.set_ylabel("R² (Test)")
ax.set_title("R² Test: Modelo Base vs Optimizado")
ax.legend()
ax.set_ylim(0, 1)

# RMSE comparativa
ax2 = axes[1]
ax2.bar(x - width/2, base["rmse_test"].values / 1000, width, label="Base", color="#357083")
ax2.bar(x + width/2, optim["rmse_test"].values / 1000, width, label="Optimizado", color="#9AA5AE")
ax2.set_xticks(x)
ax2.set_xticklabels(base["model_name"].values, rotation=20, ha="right")
ax2.set_ylabel("RMSE (miles $)")
ax2.set_title("RMSE Test: Modelo Base vs Optimizado")
ax2.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig05_model_comparison.png", dpi=150)
plt.close()
print(f"Guardada: {FIGURES_DIR / 'fig05_model_comparison.png'}")
