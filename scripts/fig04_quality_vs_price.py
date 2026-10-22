"""
Script independiente: Boxplot Calidad General vs SalePrice
Genera: outputs/figures/fig04_quality_vs_price.png
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

df = pd.read_csv("dataset.csv")

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x="Overall_Qual", y="Sale_Price", palette=sns.light_palette("#007B92", as_cmap=False), ax=ax)
ax.set_title("Precio de venta por Calidad General del inmueble", fontsize=13)
ax.set_xlabel("Overall Quality (1=Muy Malo … 10=Excelente)")
ax.set_ylabel("Precio ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
fig.suptitle(
    "INTERPRETACIÓN: La calidad general muestra una relación monotónica\n"
    "clara con el precio. Las casas de calidad 10 presentan alta varianza,\n"
    "lo que justifica el riesgo de outliers identificado por el Business Architect.",
    fontsize=8, style="italic", color="#9AA5AE"
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig04_quality_vs_price.png", dpi=150)
plt.close()
print(f"Guardada: {FIGURES_DIR / 'fig04_quality_vs_price.png'}")
