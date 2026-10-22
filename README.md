# ames-housing-ml-pipeline

Pipeline de Machine Learning para predicción de precios de vivienda construido
sobre el dataset Ames Housing. Entrena y compara 6 modelos con optimización de
hiperparámetros, coordinados mediante 5 módulos Python especializados que siguen
una metodología estructurada de extremo a extremo.

Desarrollado como trabajo del Curso de Especialización en IA y Big Data
— Instituto Nebrija (2026).

## El problema que resuelve

Fondos de inversión, bancos hipotecarios y empresas PropTech necesitan valoraciones
automáticas de propiedades con coste mínimo y alta precisión. Este proyecto evalúa
esa capacidad predictiva sobre el dataset Ames Housing y contrasta los resultados
contra dos KPIs de negocio: R² ≥ 0.85 y RMSE ≤ $25.000.

## Dataset

**Ames Housing** — OpenML ID 41211
2.930 propiedades residenciales · 81 variables · precios en USD
Mercado inmobiliario de Ames, Iowa (2006–2010)

Referencia: De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing
Data as an End of Year Statistics Project.*
Journal of Statistics Education, 19(3).

`dataset.csv` (1,4 MB) está incluido en el repositorio. También puede descargarse
ejecutando `python load_data.py`.

## Stack

| Área | Tecnología |
|---|---|
| Lenguaje | Python >= 3.12 |
| Datos | pandas >= 3.0.2, numpy >= 2.4.4, OpenML API |
| Visualización | matplotlib >= 3.10.8, seaborn >= 0.13.2 |
| Modelos ML | scikit-learn >= 1.8.0, XGBoost >= 3.2.0, LightGBM >= 4.6.0 |
| Interpretabilidad | shap >= 0.51.0 |
| Documentación | python-docx >= 1.2.0, joblib >= 1.5.3 |

## Arquitectura — 5 módulos especializados

Los "agentes" son módulos Python con responsabilidades claramente delimitadas,
coordinados por `orchestrator.py`. No usan ningún LLM externo ni API de IA;
toda la lógica es Python puro con scikit-learn y XGBoost. Cada módulo recibe
un diccionario `context` del módulo anterior y lo enriquece con sus resultados.

```
orchestrator.py
│
├── agent_01_business_architect.py  → Contexto de negocio, stakeholders y KPIs
├── agent_02_data_scientist.py      → EDA, detección de outliers y preprocesamiento
├── agent_03_ml_engineer.py         → 6 modelos, validación cruzada k-fold, RandomizedSearchCV
│       ↑                               (LinearRegression, DecisionTree, RandomForest,
│       │                                GradientBoosting, XGBoost, StackingEnsemble)
│       └── agent_04_quality_analyst.py → Auditoría: R², overfitting, RMSE
│               Umbral overfitting: gap train/test > 5 % → re-envía al ML Engineer
│               Máximo 2 ciclos de re-optimización
└── agent_05_technical_writer.py    → Memoria técnica DOCX (~16 pp) + presentación Markdown
```

## Resultados obtenidos

| Métrica | XGBoost optimizado | KPI objetivo | Estado |
|---|---|---|---|
| R² en test | 0.8837 | ≥ 0.85 | Cumple |
| RMSE en test | $19.932 | ≤ $25.000 | Cumple |
| MAE en test | $12.993 | — | — |
| Overfitting (gap R²) | 8,74 % | ≤ 5 % | No cumple |

> El auditor de calidad rechazó el modelo en el segundo ciclo porque el gap de
> overfitting (8,74 %) superó el umbral del 5 %. Al alcanzar el máximo de ciclos
> permitidos, el pipeline continuó con el mejor resultado disponible. Los KPIs
> de negocio (R² y RMSE) sí se alcanzaron.

Checks de proceso (`05_validation_report.json`): **4/4 superados**
(≥ 5 modelos entrenados, cross-validation + búsqueda de hiperparámetros,
memoria técnica 15–25 páginas, métricas R²/RMSE/MAE documentadas)

`MEMORIA_TECNICA.docx` (~1 GB) no está incluida en el repositorio por tamaño;
se regenera ejecutando el pipeline completo.

Tiempo total de ejecución en la ejecución registrada: ~117 minutos

## Cómo ejecutarlo

```bash
# 1. Crear y activar entorno virtual (requiere Python >= 3.12)
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (Opcional) Actualizar dataset desde OpenML
python load_data.py

# 4. Ejecutar el pipeline completo (~2 horas en hardware estándar)
python orchestrator.py
```

Los modelos entrenados se guardan en `outputs/models/` (excluidos del repositorio
por tamaño). Las figuras incluidas en `outputs/figures/` son las generadas durante
la ejecución original.

## Figuras incluidas

El pipeline genera 11 gráficas en `outputs/figures/`:
distribución de precios de venta, correlaciones con la variable objetivo, heatmap
de correlaciones, calidad de acabados vs precio, comparación de modelos, residuos
de LinearRegression y XGBoost, segmentos de error de ambos modelos, importancia
de variables (feature importance) y análisis SHAP de interpretabilidad local y global.

## Qué demuestra este proyecto

- Diseño de un pipeline ML estructurado en fases con separación de responsabilidades
- Bucle de control de calidad con re-optimización automática entre módulos
- Comparación de 6 familias de modelos con validación cruzada k-fold y RandomizedSearchCV
- Interpretabilidad con SHAP values
- Generación automática de documentación técnica en DOCX y Markdown

## Sobre el código

Proyecto desarrollado durante el curso de especialización. Utilicé herramientas de IA
como asistente de programación; el diseño de la arquitectura de módulos, la selección
de modelos y los criterios de auditoría son decisión mía.

## Autor

Victor Castillo Jiménez
