"""
Script independiente: Distribución de SalePrice (Original vs Log)
Genera: outputs/figures/fig01_sale_price_distribution.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cargar datos
df = pd.read_csv("dataset.csv")
df["SalePrice_log"] = np.log1p(df["Sale_Price"])

# Generar gráfica
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["Sale_Price"], kde=True, ax=axes[0], color="#007B92")
axes[0].set_title("SalePrice — Distribución Original")
axes[0].set_xlabel("Precio ($)")

sns.histplot(df["SalePrice_log"], kde=True, ax=axes[1], color="#3DB9D3")
axes[1].set_title("log(SalePrice+1) — Distribución Corregida")
axes[1].set_xlabel("log(Precio)")

fig.suptitle(
    "INTERPRETACIÓN: El sesgo positivo de SalePrice se corrige con la\n"
    "transformación logarítmica, normalizando la distribución del target.",
    fontsize=9, style="italic", color="#9AA5AE"
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig01_sale_price_distribution.png", dpi=150)
plt.close()
print(f"Guardada: {FIGURES_DIR / 'fig01_sale_price_distribution.png'}")
