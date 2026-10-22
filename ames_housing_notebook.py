# %% [markdown]
# # Ames Housing Price Prediction - Grado Institucional
# Este cuaderno concentra todo el pipeline de Análisis, Ingeniería de Características, Modelado Predictivo e Interpretabilidad Financiera desarrollado para la consultora Ames Insights.

# %% [markdown]
# ## 1. Importación de Librerías

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import joblib

# Sklearn & Models
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import xgboost as xgb
import shap

# Configuración Visual
sns.set_theme(style="whitegrid")
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 6)

# Directorios relativos al proyecto
BASE_DIR = Path()
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# %% [markdown]
# ## 2. Carga de Datos

# %%
df = pd.read_csv("dataset.csv")
print(f"Dimensiones Originales: {df.shape}")
df.head()

# %% [markdown]
# ## 3. Análisis Exploratorio de Datos (EDA) y Gráficas
# A continuación, generaremos visualizaciones para el Análisis Univariante y Multivariante relacionadas con el Precio de Venta (`Sale_Price`).

# %%
# 3.1 Distribución del Precio de Venta (Original vs Log)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Distribución original
sns.histplot(df["Sale_Price"], kde=True, ax=axes[0], color="#007B92")
axes[0].set_title("Distribución Sale_Price (Original)")
axes[0].set_xlabel("Precio ($)")

# Distribución Logarítmica
df["Log_Sale_Price"] = np.log1p(df["Sale_Price"])
sns.histplot(df["Log_Sale_Price"], kde=True, ax=axes[1], color="#3DB9D3")
axes[1].set_title("Distribución log(Sale_Price + 1)")
axes[1].set_xlabel("log(Precio)")

plt.tight_layout()
plt.show()

# %%
# 3.2 Top Correlaciones con el Precio
corr_matrix = df.select_dtypes(include=[np.number]).corr()
correlations = corr_matrix["Sale_Price"].sort_values(ascending=False)
top_corr = correlations[1:16]  # Excluir self-correlation

plt.figure(figsize=(10, 6))
sns.barplot(x=top_corr.values, y=top_corr.index, palette=["#357083", "#007B92", "#3DB9D3"])
plt.title("Top 15 Correlaciones Positivas con Sale_Price")
plt.xlabel("Coeficiente de Correlación de Pearson")
plt.show()

# %%
# 3.3 Visualización de Calidad Inmobiliaria (Predictor #1)
plt.figure(figsize=(10, 6))
sns.boxplot(x="Overall_Qual", y="Sale_Price", data=df, palette=sns.blend_palette(["#357083", "#007B92", "#3DB9D3"], as_cmap=False))
plt.title("Impacto de la Calidad General de Acabados en el Precio")
plt.xlabel("Calidad General (1 a 10)")
plt.ylabel("Precio de Venta ($)")
plt.show()

# %% [markdown]
# ## 4. Preprocesamiento e Ingeniería de Características Avanzadas
# Basado en la exploración, procedemos a curar el dataset para el Modelado.

# %%
def preprocess_ames(df_input):
    df_clean = df_input.copy()
    
    # Estandarización de nombres de columnas
    df_clean.columns = df_clean.columns.str.replace(" ", "_").str.replace("/", "_")
    
    # Limpieza de nulos por conocimiento de negocio
    cols_none = ["Alley", "Mas_Vnr_Type", "Bsmt_Qual", "Bsmt_Cond", "Bsmt_Exposure", 
                 "BsmtFin_Type_1", "BsmtFin_Type_2", "Fireplace_Qu", "Garage_Type", 
                 "Garage_Finish", "Garage_Qual", "Garage_Cond", "Pool_QC", "Fence", "Misc_Feature"]
    for c in cols_none:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].fillna("None")
            
    # Imputación Numérica
    for c in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[c].isnull().any():
            if "Garage" in c or "Bsmt" in c or "Mas_Vnr_Area" in c:
                df_clean[c] = df_clean[c].fillna(0)
            else:
                df_clean[c] = df_clean[c].fillna(df_clean[c].median())
                
    # Imputación Categórica
    for c in df_clean.select_dtypes(include=["object"]).columns:
        if df_clean[c].isnull().any():
            df_clean[c] = df_clean[c].fillna(df_clean[c].mode()[0])
            
    # Filtro de Outliers
    df_clean = df_clean[df_clean["Gr_Liv_Area"] <= 4000]
    
    # ------ FEATURE ENGINEERING (V2 Institucional) ------
    if "Gr_Liv_Area" in df_clean.columns and "Total_Bsmt_SF" in df_clean.columns:
        df_clean["Pct_Finished_Area"] = df_clean["Gr_Liv_Area"] / (df_clean["Gr_Liv_Area"] + df_clean["Total_Bsmt_SF"] + 1)
        
    if "Overall_Qual" in df_clean.columns and "Gr_Liv_Area" in df_clean.columns:
        df_clean["Qual_x_Area"] = df_clean["Overall_Qual"] * df_clean["Gr_Liv_Area"]
        
    if "Yr_Sold" in df_clean.columns and "Year_Remod_Add" in df_clean.columns:
        df_clean["Years_Since_Remod"] = df_clean["Yr_Sold"] - df_clean["Year_Remod_Add"]
        
    # Variables Redundantes y Target Original
    cols_to_drop = ["Order", "PID", "Log_Sale_Price"]
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])
    
    return df_clean

df_processed = preprocess_ames(df)

# Vectorización
y = np.log1p(df_processed["Sale_Price"])
X = df_processed.drop("Sale_Price", axis=1)
X = pd.get_dummies(X, drop_first=True)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

# %% [markdown]
# ## 5. Entrenamiento de Modelos (Base y Stacking)
# Probamos distintas geometrías algorítmicas, destacando el **StackingRegressor**.

# %%
# Definimos Base Estimators para Stacking
base_estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0))
]

# Definimos Diccionario de Modelos a probar
modelos = {
    "Ridge Regression": Ridge(alpha=10),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0),
    "Stacking Ensemble": StackingRegressor(estimators=base_estimators, final_estimator=Ridge())
}

# Entrenamiento Rápido
metricas = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    r2_test = modelo.score(X_test, y_test)
    rmse_test = np.sqrt(np.mean((np.expm1(y_test) - np.expm1(y_pred))**2))
    metricas[nombre] = {"R2": r2_test, "RMSE": rmse_test}

# Imprimir Resultados
df_metricas = pd.DataFrame(metricas).T
print("Resultados de Modelado en Test Set:\n")
try:
    display(df_metricas.style.format({"R2": "{:.4f}", "RMSE": "${:,.0f}"}).background_gradient(cmap="viridis"))
except NameError:
    print(df_metricas)

# %% [markdown]
# ## 6. Interpretabilidad SHAP (XGBoost)
# Demostremos cómo opera el motor interno permitiendo visualizar el recibo de decisión en una predicción individual.

# %%
mejor_modelo = modelos["XGBoost"]  # Extraemos para análisis SHAP debido a soporte nativo

# Extracción global (Summary Plot)
explainer = shap.TreeExplainer(mejor_modelo)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Ames Housing - Top Drivers Globales del Valor (XGBoost)")
plt.show()

# %% [markdown]
# ### Análisis Local (Gráfico de Cascada o Waterfall)
# Evaluamos por qué una única casa fue cotizada a su correspondiente valor.

# %%
indice_vivienda_ejemplo = 0
plt.figure()
shap.plots.waterfall(shap_values[indice_vivienda_ejemplo], max_display=10, show=False)
plt.title(f"Recibo Matemático: Desglose de Predicción Vivienda ID={indice_vivienda_ejemplo}")
try:
    if not __file__: pass
    print("La gráfica de cascada (Waterfall) de SHAP se ha generado.")
except NameError:
    plt.show()

print("\n✅ PIPELINE COMPLETADO EXITOSAMENTE")
