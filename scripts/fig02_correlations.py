"""
Script independiente: Top 15 Correlaciones con SalePrice
Genera: outputs/figures/fig02_correlations.png
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
top_pos = corr.head(10)
top_neg = corr.tail(5)
top_corr = pd.concat([top_pos, top_neg])

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#007B92" if v > 0 else "#357083" for v in top_corr.values]
top_corr.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Top 15 Correlaciones con SalePrice", fontsize=13)
ax.set_xlabel("Coeficiente de Pearson")
ax.axvline(0, color="black", linewidth=0.8)
fig.suptitle(
    "INTERPRETACIÓN: Gr_Liv_Area (superficie habitable) y Overall_Qual\n"
    "(calidad general) son los predictores más fuertes. Variables como\n"
    "BsmtFin_SF_2 muestran correlación negativa — indicio de multicolinealidad.",
    fontsize=8, style="italic", color="#9AA5AE"
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig02_correlations.png", dpi=150)
plt.close()
print(f"Guardada: {FIGURES_DIR / 'fig02_correlations.png'}")
