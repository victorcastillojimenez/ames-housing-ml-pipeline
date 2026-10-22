"""
Script independiente: Análisis de Interpretabilidad SHAP (Local y Global)
Genera: outputs/figures/fig09_shap_analysis.png
Requiere: outputs/models/<modelo>_optimized.pkl, outputs/X_test.csv
          y la librería `shap` instalada (`pip install shap`)
"""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Suprimir avisos de SHAP
warnings.filterwarnings("ignore")

try:
    import shap
except ImportError:
    print("Error: La librería 'shap' no está instalada. Ejecuta: pip install shap")
    exit(1)

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Leer mejor modelo
with open("outputs/04_quality_report.json", "r", encoding="utf-8") as f:
    qa = json.load(f)
best_name = qa["best_model_name"]

print(f"Generando SHAP analysis para el modelo: {best_name}")

# Cargar modelo y datos (usar subset para acelerar KernelExplainer si aplica)
model = joblib.load(f"outputs/models/{best_name}_optimized.pkl")
X_train = pd.read_csv("outputs/X_train.csv")
X_test = pd.read_csv("outputs/X_test.csv")

# Usar el set escalado si es Lineal
if "Linear" in best_name or "Ridge" in best_name:
    X_train = pd.read_csv("outputs/X_train_scaled.csv")
    X_test = pd.read_csv("outputs/X_test_scaled.csv")

# Tomamos una muestra de test para el BeeSwarms (global) y la primera instancia para cascada (local)
X_test_sample = X_test.sample(n=min(150, len(X_test)), random_state=42)
X_train_bg = shap.utils.sample(X_train, 50)  # Background para el explainer agnóstico

# Crear explainer
if "RandomForest" in best_name or "XGB" in best_name or "Gradient" in best_name or "DecisionTree" in best_name:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_sample)
        # Para RandomForest, shap_values a veces tiene forma (n_samples, n_features, 1). Lo corregimos.
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., 0]
    except Exception as e:
        print(f"Fallback a explainer genérico por error en TreeExplainer: {e}")
        explainer = shap.Explainer(model.predict, X_train_bg)
        shap_values = explainer(X_test_sample)
else:
    # StackingRegressor o modelos lineales
    explainer = shap.Explainer(model.predict, X_train_bg)
    shap_values = explainer(X_test_sample)

# -- Generar gráfica combinada --
fig = plt.figure(figsize=(15, 6))
plt.subplots_adjust(wspace=0.6)

# 1. SHAP Summary Plot (Global)
ax1 = fig.add_subplot(1, 2, 1)
shap.summary_plot(shap_values, X_test_sample, show=False, max_display=10, plot_size=None)
ax1.set_title("Explicabilidad Global: Impacto de variables\n(BeeSwarm Plot)", fontsize=11)

# 2. SHAP Waterfall Plot (Local) - Explicando la instancia #0 del sample
ax2 = fig.add_subplot(1, 2, 2)
# Elegimos una propiedad cara del sample para que el waterfall sea interesante
idx_expensive = y_pred_sample = model.predict(X_test_sample).argmax()
shap_val_instance = shap_values[idx_expensive]

shap.plots.waterfall(shap_val_instance, max_display=8, show=False)
ax2.set_title("Explicabilidad Local: Tasación de una Propiedad Específica\n(Waterfall Plot)", fontsize=11)

fig.suptitle(
    "INTERPRETABILIDAD SHAP PARA EL NEGOCIO (REITs & Bancos Hipotecarios)\n"
    "Izquierda: Impacto sistémico de cada feature y detección de sesgos (Global).\n"
    "Derecha: Desglose peso a peso ($) de la valoración de UNA vivienda en particular (Auditoría Local).",
    fontsize=9, style="italic", color="#9AA5AE", y=1.02
)

file_out = FIGURES_DIR / "fig09_shap_analysis.png"
plt.savefig(file_out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅  Guardada: {file_out}")
