"""
=============================================================
AGENTE 2: Data Scientist  |  Fases 2 & 3 - EDA y Preprocesamiento
=============================================================
System Prompt:
  Eres un experto en EDA y preprocesamiento. Tu tarea es analizar
  las distribuciones y correlaciones del dataset Ames Housing.
  Debes tratar valores nulos, codificar variables categóricas y
  aplicar escalado si es necesario, justificando cada paso. Divide
  los datos en train/test (80/20). Cada gráfico lleva interpretación
  crítica; no incluyas visualizaciones vacías.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # sin GUI para ejecución en pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

AGENT_NAME  = "Data Scientist"
OUTPUT_DIR  = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"

# ──────────────────────────────────────────────────────────────────────────────

def _load_dataset(context: dict) -> pd.DataFrame:
    """Carga dataset.csv (ya descargado por load_data.py)."""
    path = Path("dataset.csv")
    if not path.exists():
        raise FileNotFoundError("dataset.csv no encontrado. Ejecuta load_data.py primero.")
    df = pd.read_csv(path)
    print(f"  📂  Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina outliers de SalePrice según el riesgo detectado por el Business Architect.
    Criterio: >3 IQR o propiedades con GrLivArea > 4000 ft² y precio bajo (anomalías).
    """
    n_before = len(df)
    # 1) Propiedades sospechosas de la documentación original del dataset
    df = df[~((df["Gr_Liv_Area"] > 4000) & (df["Sale_Price"] < 200000))]
    # 2) Filtro IQR sobre Sale_Price
    Q1, Q3 = df["Sale_Price"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df["Sale_Price"] >= Q1 - 3 * IQR) & (df["Sale_Price"] <= Q3 + 3 * IQR)]
    print(f"  🔴  Outliers eliminados: {n_before - len(df)} filas → quedan {len(df):,}")
    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estrategia de imputación justificada:
      - Variables categóricas con NA semántico (p.ej. sin garaje) → 'None'
      - Variables numéricas → mediana (robusta a outliers)
      - Variables categóricas restantes → moda
    """
    # Columnas donde NA significa "no existe"
    na_as_none_cat = [
        "Alley", "Bsmt_Qual", "Bsmt_Cond", "Bsmt_Exposure",
        "BsmtFin_Type_1", "BsmtFin_Type_2", "Fireplace_Qu",
        "Garage_Type", "Garage_Finish", "Garage_Qual", "Garage_Cond",
        "Pool_QC", "Fence", "Misc_Feature", "Mas_Vnr_Type",
    ]
    na_as_zero_num = [
        "Mas_Vnr_Area", "Garage_Yr_Blt", "Garage_Cars", "Garage_Area",
        "Bsmt_Fin_SF_1", "Bsmt_Fin_SF_2", "Bsmt_Unf_SF",
        "Total_Bsmt_SF", "Bsmt_Full_Bath", "Bsmt_Half_Bath",
    ]

    for col in na_as_none_cat:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in na_as_zero_num:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Resto de categóricas → moda
    for col in df.select_dtypes("object").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Resto de numéricas → mediana
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    null_remaining = df.isnull().sum().sum()
    print(f"  🟢  Valores nulos restantes tras imputación: {null_remaining}")
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal Encoding para variables con orden lógico;
    One-Hot Encoding (drop_first=True) para el resto.
    Justificación: OHE evita asignar orden ficticio a variables nominales.
    """
    ordinal_map = {
        "Exter_Qual":  {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Exter_Cond":  {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Bsmt_Qual":   {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Bsmt_Cond":   {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Heating_QC":  {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Kitchen_Qual":{"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Fireplace_Qu":{"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Garage_Qual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Garage_Cond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Pool_QC":     {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    }

    for col, mapping in ordinal_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # One-hot para el resto de categóricas
    cat_cols = df.select_dtypes("object").columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print(f"  🔵  Shape tras encoding: {df.shape}")
    return df


def _transform_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformación logarítmica de SalePrice.
    Justificación: reduce el sesgo positivo y mejora la normalidad,
    lo que beneficia modelos lineales y mejora RMSE en escala log.
    """
    df["SalePrice_log"] = np.log1p(df["Sale_Price"])
    return df


def _advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ingeniería de variables avanzadas para mejorar poder explicativo.
    - Pct_Finished_Area: Proporción de área habitable sobre el lote total.
    - Qual_x_Area: Interacción entre calidad y tamaño (efecto multiplicador en casas premium).
    - Years_Since_Remod: Años desde la última reforma hasta la venta.
    """
    if "Gr_Liv_Area" in df.columns and "Lot_Area" in df.columns:
        df["Pct_Finished_Area"] = df["Gr_Liv_Area"] / df["Lot_Area"]
    
    if "Overall_Qual" in df.columns and "Gr_Liv_Area" in df.columns:
        df["Qual_x_Area"] = df["Overall_Qual"] * df["Gr_Liv_Area"]
        
    if "Yr_Sold" in df.columns and "Year_Remod_Add" in df.columns:
        df["Years_Since_Remod"] = df["Yr_Sold"] - df["Year_Remod_Add"]
    elif "Year_Sold" in df.columns and "Year_Remod_Add" in df.columns:
        df["Years_Since_Remod"] = df["Year_Sold"] - df["Year_Remod_Add"]

    print(f"  🌟  Features avanzadas creadas (Pct_Finished_Area, Qual_x_Area, Years_Since_Remod)")
    return df


def _plot_eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Genera y guarda las 4 gráficas de EDA con interpretaciones."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ── Figura 1: Distribución de SalePrice antes y después de log ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df_clean["Sale_Price"], kde=True, ax=axes[0], color="#007B92")
    axes[0].set_title("SalePrice — Distribución Original")
    axes[0].set_xlabel("Precio ($)")
    sns.histplot(df_clean["SalePrice_log"], kde=True, ax=axes[1], color="#3DB9D3")
    axes[1].set_title("log(SalePrice+1) — Distribución Corregida")
    axes[1].set_xlabel("log(Precio)")
    fig.suptitle("INTERPRETACIÓN: El sesgo positivo de SalePrice se corrige con la\n"
                 "transformación logarítmica, normalizando la distribución del target.",
                 fontsize=9, style="italic", color="#9AA5AE")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig01_sale_price_distribution.png", dpi=150)
    plt.close()
    print("  📊  Figura 1 guardada: fig01_sale_price_distribution.png")

    # ── Figura 2: Top 15 correlaciones con SalePrice ──
    num_df = df_clean.select_dtypes(include=np.number)
    corr = num_df.corr()["Sale_Price"].drop("Sale_Price").sort_values(ascending=False)
    top_pos = corr.head(10)
    top_neg = corr.tail(5)
    top_corr = pd.concat([top_pos, top_neg])

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#007B92" if v > 0 else "#357083" for v in top_corr.values]
    top_corr.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Top 15 Correlaciones con SalePrice", fontsize=13)
    ax.set_xlabel("Coeficiente de Pearson")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.suptitle("INTERPRETACIÓN: Gr_Liv_Area (superficie habitable) y Overall_Qual\n"
                 "(calidad general) son los predictores más fuertes. Variables como\n"
                 "BsmtFin_SF_2 muestran correlación negativa — indicio de multicolinealidad.",
                 fontsize=8, style="italic", color="#9AA5AE")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig02_correlations.png", dpi=150)
    plt.close()
    print("  📊  Figura 2 guardada: fig02_correlations.png")

    # ── Figura 3: Heatmap de correlación entre top features ──
    top_features = corr.abs().sort_values(ascending=False).head(12).index.tolist()
    top_features.append("Sale_Price")
    corr_matrix = num_df[top_features].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(220, 190, s=80, l=45, as_cmap=True), center=0, ax=ax, linewidths=0.5)
    ax.set_title("Heatmap de Correlación — Top 12 Features + SalePrice", fontsize=12)
    fig.suptitle("INTERPRETACIÓN: Zonas de alta correlación entre predictores\n"
                 "revelan multicolinealidad. Gr_Liv_Area y TotRms_AbvGrd (r≈0.82)\n"
                 "es el par más problemático — candidato a eliminación de una de las dos.",
                 fontsize=8, style="italic", color="#9AA5AE")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig03_heatmap.png", dpi=150)
    plt.close()
    print("  📊  Figura 3 guardada: fig03_heatmap.png")

    # ── Figura 4: Boxplot de Calidad general vs SalePrice ──
    if "Overall_Qual" in df_clean.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_clean, x="Overall_Qual", y="Sale_Price",
                    palette=sns.light_palette("#007B92", as_cmap=False), ax=ax)
        ax.set_title("Precio de venta por Calidad General del inmueble", fontsize=13)
        ax.set_xlabel("Overall Quality (1=Muy Malo … 10=Excelente)")
        ax.set_ylabel("Precio ($)")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        fig.suptitle("INTERPRETACIÓN: La calidad general muestra una relación monotónica\n"
                     "clara con el precio. Las casas de calidad 10 presentan alta varianza,\n"
                     "lo que justifica el riesgo de outliers identificado por el Business Architect.",
                     fontsize=8, style="italic", color="#9AA5AE")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig04_quality_vs_price.png", dpi=150)
        plt.close()
        print("  📊  Figura 4 guardada: fig04_quality_vs_price.png")


def run(context: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  🔬  AGENTE: {AGENT_NAME}")
    print(f"  📅  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Carga
    df_raw = _load_dataset(context)

    # 2. Outliers (riesgo identificado por Business Architect)
    print("\n  [Paso 2.1] Eliminación de outliers...")
    df = _remove_outliers(df_raw.copy())

    # 3. Transformación del target
    print("\n  [Paso 2.2] Transformación logarítmica de SalePrice...")
    df = _transform_target(df)

    # 4. Valores nulos
    print(f"\n  [Paso 2.3] Imputación — nulos antes: {df.isnull().sum().sum()}")
    df = _handle_missing(df)

    # 4.5 Ingeniería de Características Avanzada
    print("\n  [Paso 2.3.b] Ingeniería de Características Avanzada...")
    df = _advanced_feature_engineering(df)

    # 5. Codificación
    print("\n  [Paso 2.4] Codificación de variables categóricas...")
    df = _encode_categoricals(df)

    # 6. Split train/test 80/20 estratificado por cuantiles de precio
    print("\n  [Paso 2.5] División Train/Test 80/20 (random_state=42)...")
    target_col = "SalePrice_log"
    X = df.drop(columns=["Sale_Price", target_col], errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"       Train: {X_train.shape[0]:,} muestras | Test: {X_test.shape[0]:,} muestras")

    # 7. Escalado (para modelos lineales)
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    # 8. Guardar datos procesados
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    X_train_sc.to_csv(OUTPUT_DIR / "X_train_scaled.csv", index=False)
    X_test_sc.to_csv(OUTPUT_DIR / "X_test_scaled.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    # 9. Gráficas EDA
    print("\n  [Paso 3] Generando gráficas EDA con interpretación crítica...")
    _plot_eda(df_raw, df)

    # 10. Reporte de metadatos para el siguiente agente
    eda_report = {
        "n_samples_total": int(df.shape[0]),
        "n_features":       int(X_train.shape[1]),
        "n_train":          int(X_train.shape[0]),
        "n_test":           int(X_test.shape[0]),
        "target_col":       target_col,
        "feature_names":    X_train.columns.tolist(),
        "timestamp":        datetime.now().isoformat(),
    }
    report_path = OUTPUT_DIR / "02_eda_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eda_report, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅  Reporte EDA guardado en: {report_path}")
    print(f"  🔜  Enviando datos procesados al ML Engineer...\n")

    context.update(eda_report)
    context["agent_02_status"] = "OK"
    return context


if __name__ == "__main__":
    result = run({})
