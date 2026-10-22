"""
=============================================================
AGENTE 1: Business Architect  |  Fase 1 - Contexto de Negocio
=============================================================
System Prompt:
  Eres el Director Estratégico de una consultora de IA. Tu misión es
  definir el contexto empresarial para el dataset Ames Housing
  (predicción de precios de viviendas). Identifica riesgos como la
  multicolinealidad o valores atípicos que afecten al negocio.
"""

import json
import os
from datetime import datetime
from pathlib import Path

AGENT_NAME = "Business Architect"
OUTPUT_DIR  = Path("outputs")

# ---------------------------------------------------------------------------
def run(context: dict) -> dict:
    """
    Genera el documento de Contexto y Objetivos del proyecto.
    Retorna un dict que será pasado al siguiente agente.
    """
    print(f"\n{'='*60}")
    print(f"  🏛️  AGENTE: {AGENT_NAME}")
    print(f"  📅  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    business_context = {
        "empresa": "Ames Insights Consulting",
        "lema": "Data-Driven Decisions for the Future of Real Estate",
        "objetivo_negocio": (
            "Desarrollar un modelo predictivo de precios de vivienda para el "
            "mercado de Ames, Iowa, que permita a fondos de inversión, bancos "
            "hipotecarios y empresas PropTech obtener valoraciones automáticas "
            "con error inferior al 10 % del precio real."
        ),
        "stakeholders": [
            "Gestores de Fondos Inmobiliarios (REITs)",
            "Bancos Hipotecarios",
            "Empresas PropTech",
        ],
        "impacto_economico": (
            "Ahorro Estructural: Un error del 5 % en una propiedad de $300,000 "
            "equivale a $15,000 de riesgo directo. Reemplazar una tasación manual "
            "tradicional ($400/unidad) por nuestra inferencia algorítmica ($0.10/unidad) "
            "generaría un impacto directo de ~$4 millones anuales para un portfolio "
            "de 10,000 operaciones, acelerando además la aprobación hipotecaria de semanas a minutos."
        ),
        "contexto_macroeconomico": [
            {
                "nombre": "Resiliencia ante la Gran Recesión (2008)",
                "descripcion": "El dataset registra ventas entre 2006 y 2010. A diferencia de las caídas catastróficas del 30-50% en zonas costeras, el mercado de Middle America (Ames) demostró alta resiliencia. El inversor debe notar esta estabilidad como mitigador de riesgo sistemático."
            },
            {
                "nombre": "Limitación de Inflación",
                "descripcion": "El modelo no contempla la inflación posterior a 2010. Para despliegues en producción de grado institucional, proponemos integrar el Índice de Precios de Vivienda (HPI) actual como multiplicador macro al output nominal del algoritmo."
            }
        ],
        "riesgos_identificados": [
            {
                "nombre": "Valores Atípicos (Outliers)",
                "descripcion": (
                    "Propiedades de lujo con características únicas pueden sesgar "
                    "el modelo. Acción: detectar y gestionar outliers en SalePrice."
                ),
            },
            {
                "nombre": "Multicolinealidad",
                "descripcion": (
                    "Variables como 'GrLivArea' y 'TotRmsAbvGrd' están altamente "
                    "correlacionadas. Acción: análisis VIF y selección de features."
                ),
            },
            {
                "nombre": "Datos Faltantes",
                "descripcion": (
                    "Múltiples columnas tienen valores nulos (Pool, Garage, Basement). "
                    "Acción: estrategia de imputación documentada y justificada."
                ),
            },
            {
                "nombre": "Distribución Sesgada del Target",
                "descripcion": (
                    "SalePrice tiene sesgo positivo. Acción: aplicar transformación "
                    "logarítmica para mejorar la normalidad y el desempeño del modelo."
                ),
            },
        ],
        "kpis_exito": {
            "R2_minimo": 0.85,
            "RMSE_maximo": 25000,
            "MAE_maximo": 18000,
            "umbral_overfitting": 0.05,   # diferencia máx. R² train vs test
        },
        "dataset_id": 41211,
        "target_variable": "SalePrice",
        "timestamp": datetime.now().isoformat(),
    }

    # ---- Guardar reporte ----
    report_path = OUTPUT_DIR / "01_business_context.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(business_context, f, ensure_ascii=False, indent=2)

    # ---- Imprimir resumen ----
    print(f"\n  🏢  Empresa  : {business_context['empresa']}")
    print(f"  🎯  Objetivo : {business_context['objetivo_negocio'][:80]}...")
    print(f"\n  ⚠️   Riesgos identificados:")
    for r in business_context["riesgos_identificados"]:
        print(f"       • {r['nombre']}")
    print(f"\n  📊  KPIs mínimos aceptados:")
    kpis = business_context["kpis_exito"]
    print(f"       • R² ≥ {kpis['R2_minimo']}")
    print(f"       • RMSE ≤ ${kpis['RMSE_maximo']:,}")
    print(f"       • MAE  ≤ ${kpis['MAE_maximo']:,}")
    print(f"       • Overfitting máximo: {kpis['umbral_overfitting'] * 100:.1f}% de diferencia R²")
    print(f"\n  ✅  Reporte guardado en: {report_path}")
    print(f"\n  🔜  Enviando contexto al Data Scientist...\n")

    # Propagar contexto al siguiente agente
    context.update(business_context)
    context["agent_01_status"] = "OK"
    return context


if __name__ == "__main__":
    result = run({})
    print(json.dumps(result, ensure_ascii=False, indent=2))
