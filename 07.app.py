# -*- coding: utf-8 -*-
"""
APLICACIÓN DE TRIAJE (PROTOTIPO) — Streamlit
=============================================
Ejecución (en la carpeta del proyecto, con los .joblib en outputs/models/):
    streamlit run app_triage.py

Nota: el umbral por defecto corresponde al punto operativo interno
(sensibilidad >= 90 %). En un despliegue clínico real debe recalibrarse local.
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "outputs", "models")

FEATURES_20 = [
    "HBA1C", "PCV", "LEADDP", "SCR", "DBILI", "BU", "CHD",
    "ENDOCRINE_DISEASE", "ALB", "GGT", "Systolic blood pressure", "AGE",
    "TP", "PT", "TBILI", "PLT", "BMI", "GLU", "Diastolic blood pressure",
    "gender",
]
FEATURES_17 = [f for f in FEATURES_20 if f not in ("DBILI", "PT", "LEADDP")]
BINARIAS = {"LEADDP", "CHD", "ENDOCRINE_DISEASE", "gender"}

# Rangos clínicamente plausibles para validar la entrada del formulario.
REF = {
    "AGE": (18, 100), "HBA1C": (3, 20), "GLU": (2, 40), "BMI": (12, 60),
    "Systolic blood pressure": (70, 260), "Diastolic blood pressure": (40, 150),
    "SCR": (20, 1500), "BU": (0.5, 120), "ALB": (10, 60), "TP": (30, 100),
    "TBILI": (1, 350), "DBILI": (0, 300), "GGT": (3, 1500), "PCV": (12, 75),
    "PLT": (2, 1400), "PT": (9, 150),
}

MODELOS = {
    "Principal (20 biomarcadores)": ("catboost.joblib", FEATURES_20),
    "Transportable (17 biomarcadores)": ("catboost_transportable.joblib", FEATURES_17),
}

st.set_page_config(page_title="Triaje de Retinopatía Diabética", layout="wide")
st.title("Triaje de retinopatía diabética")
st.caption("Herramienta de apoyo a la decisión: estima la probabilidad de "
           "retinopatía diabética a partir de biomarcadores de laboratorio de "
           "rutina y recomienda derivar o mantener seguimiento.")

with st.sidebar:
    st.header("Configuración")
    nombre = st.selectbox("Modelo", list(MODELOS.keys()))
    umbral = st.number_input("Umbral de derivación", 0.0, 1.0, 0.2524, 0.01,
                             help="Punto operativo interno (sensibilidad >= 90 %). "
                                  "")

archivo, features = MODELOS[nombre]
ruta = os.path.join(MODEL_DIR, archivo)
if not os.path.exists(ruta):
    st.error(f"No se encontró el modelo en {ruta}. Ejecuta primero "
             f"03_train_models.py (y 06 para el transportable).")
    st.stop()
modelo = joblib.load(ruta)

st.header("Datos del paciente")
cols = st.columns(4)
valores = {}
for i, f in enumerate(features):
    with cols[i % 4]:
        if f in BINARIAS:
            valores[f] = float(st.selectbox(f, [0, 1], key=f))
        else:
            lo, hi = REF.get(f, (0, 1000))
            lo = float(lo)
            hi = float(hi)
            med = (lo + hi) / 2.0
            valores[f] = st.number_input(
                f,
                min_value=lo,
                max_value=hi,
                value=med,
                key=f
            )

if st.button("Calcular riesgo", type="primary"):
    X = pd.DataFrame([valores])[features].astype(float)
    prob = float(modelo.predict_proba(X)[0, 1])
    derivar = prob >= umbral

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de RD", f"{prob:.3f}")
    c2.metric("Umbral", f"{umbral:.3f}")
    c3.metric("Decisión", "DERIVAR" if derivar else "SEGUIMIENTO")
    if derivar:
        st.warning("Riesgo alto: priorizar derivación a oftalmología.")
    else:
        st.success("Riesgo bajo: mantener seguimiento rutinario. "
                   "No descarta el tamizaje anual recomendado.")


    with st.expander("Explicabilidad (SHAP)", expanded=True):
        try:
            import shap
            import matplotlib.pyplot as plt
            import textwrap

            # Obtener el clasificador final del pipeline
            inner = modelo.named_steps.get("clf", modelo)
            expl = shap.TreeExplainer(inner)
            sv = expl.shap_values(X)

            # Para clasificación binaria, utilizar la clase positiva
            if isinstance(sv, list):
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]

            # Calcular importancia global mediante SHAP absoluto medio
            contrib = pd.Series(
                np.abs(sv).mean(axis=0),
                index=features,
                name="Importancia SHAP"
            )

            # Orden descendente: la variable más importante queda arriba
            contrib = contrib.sort_values(ascending=True)

            # Ajustar altura al número de variables
            n_features = len(contrib)
            fig, ax = plt.subplots(
                figsize=(11, max(6, n_features * 0.42))
            )

            # Etiquetas legibles, incluyendo nombres largos
            etiquetas = [
                "\n".join(textwrap.wrap(str(nombre), width=28))
                for nombre in contrib.index
            ]

            # Gráfico de barras horizontal
            barras = ax.barh(
                etiquetas,
                contrib.values,
                color="#2878B5",
                edgecolor="white",
                height=0.72
            )

            # Valores al final de las barras
            max_val = float(contrib.max()) if len(contrib) else 0.0
            ax.set_xlim(0, max_val * 1.18 if max_val > 0 else 1)

            for barra, valor in zip(barras, contrib.values):
                ax.text(
                    barra.get_width() + max_val * 0.015,
                    barra.get_y() + barra.get_height() / 2,
                    f"{valor:.4f}",
                    va="center",
                    ha="left",
                    fontsize=9
                )

            ax.set_title(
                "Importancia global de los biomarcadores (SHAP)",
                fontsize=14,
                fontweight="bold",
                pad=14
            )
            ax.set_xlabel("Media del valor absoluto SHAP", fontsize=11)
            ax.set_ylabel("Biomarcador", fontsize=11)

            ax.tick_params(axis="y", labelsize=10)
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)

            # Eliminar bordes visuales innecesarios
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption(
                "Las variables están ordenadas por importancia global. "
                "Un valor mayor indica una contribución absoluta media "
                "más alta en las predicciones del modelo; no indica "
                "por sí solo si el riesgo aumenta o disminuye."
            )

        except Exception as e:  # noqa: BLE001
            st.warning(f"SHAP no disponible: {e}")
