"""
Script independiente: Heatmap de Correlación — Top 12 Features + SalePrice
Genera: outputs/figures/fig03_heatmap.png
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
num_df = df.select_dtypes(include=np.number)
corr = num_df.corr()["Sale_Price"].drop("Sale_Price").sort_values(ascending=False)
top_features = corr.abs().sort_values(ascending=False).head(12).index.tolist()
top_features.append("Sale_Price")
corr_matrix = num_df[top_features].corr()

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(220, 190, s=80, l=45, as_cmap=True), center=0, ax=ax, linewidths=0.5)
ax.set_title("Heatmap de Correlación — Top 12 Features + SalePrice", fontsize=12)
fig.suptitle(
    "INTERPRETACIÓN: Zonas de alta correlación entre predictores\n"
    "revelan multicolinealidad. Gr_Liv_Area y TotRms_AbvGrd (r≈0.82)\n"
    "es el par más problemático — candidato a eliminación de una de las dos.",
    fontsize=8, style="italic", color="#9AA5AE"
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig03_heatmap.png", dpi=150)
plt.close()
print(f"Guardada: {FIGURES_DIR / 'fig03_heatmap.png'}")
