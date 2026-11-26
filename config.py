# -*- coding: utf-8 -*-
"""
================================================================================
 CONFIGURACIÓN CENTRAL — Pipeline de triaje de retinopatía diabética
 Edita SOLO este archivo para cambiar rutas, modelos, umbrales, GPU, etc.
================================================================================
"""
import os
from pathlib import Path

# ============================================================================
# 1. RUTAS
# ============================================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                    # pon aquí tu dataset.xlsx
OUTPUTS = ROOT / "outputs"

CLEAN_DIR = OUTPUTS / "clean"               # paso 01
FEAT_DIR = OUTPUTS / "features"             # paso 02
MODEL_DIR = OUTPUTS / "models"              # paso 03
EVAL_DIR = OUTPUTS / "eval"                 # paso 04
EXT_DIR = OUTPUTS / "external"              # paso 05
LOG_DIR = OUTPUTS / "logs"
FIG_DIR = EVAL_DIR / "figs"

RAW_XLSX = DATA_DIR / "dataset.xlsx"
PARSED_CSV = ROOT.parent / "audit_output" / "parsed_data.csv"

# ============================================================================
# 2. DATASET Y VALIDACIÓN
# ============================================================================
TARGET = "Disease Type"         # columna objetivo (0 = diabetes, 1 = RD)
ID_COL = "Case_ID"
SEED = 42
TEST_SIZE = 0.2                 # hold-out estratificado intocado
CV_FOLDS = 5                    # folds para tuning/estimación
N_JOBS = -1                     # -1 = usar todos los núcleos (tienes RAM de sobra)

REAL_PREVALENCE = 0.30          # prevalencia real de RD en población diabética
TARGET_SENSITIVITY = 0.90       # punto operativo de triaje (sensibilidad fijada)

# ============================================================================
# 3. LIMPIEZA (paso 01)
# ============================================================================
# Columnas eliminadas siempre (ver reporte de auditoría)
DROP_ALWAYS = [
    "BUN", "M1_M2", "TH2", "FIBRIN", "HEART_RATE", "LP_A", "PL", "IBILI",
    "Marital status", "PREGNANT", "PCOS", "BREAST_TUMOR", "UCR",
    "GLU_2H", "INS", "CP",
]
# Variables sospechosas de look-ahead (registradas durante/después del dx de RD):
# NUNCA entran al modelo final; solo se usan en análisis de sesgo.
LOOKAHEAD_EXCLUDE = ["NEPHROPATHY", "RENAL_FALIURE"]
# Tier-2 renal/inflamatorio: solo para diagnósticos de sesgo, no al modelo final.
TIER2_DIAG = ["ALB_CR", "UPR_24", "ESR", "GSP", "CRP"]

# Reglas de corrección de unidades (detectadas en la auditoría)
UNIT_FIXES = dict(
    fbg_mgdl_threshold=50.0,   # FBG > 50 => asumido mg/dL -> /18
    fbg_min_valid=2.0,         # FBG < 2  => NaN (hipoglucemia imposible)
    bu_threshold=50.0,         # BU > 50  => asumido BUN mg/dL -> /2.8
    bu_clip=120.0,
    pta_fraction_max=5.0,      # PTA < 5  => asumido fracción -> x100
    height_range=(130.0, 210.0),
    weight_range=(25.0, 250.0),
    scr_max=1500.0,
)

# ============================================================================
# 4. SELECCIÓN DE FEATURES (paso 02) — embudo multi-criterio
# ============================================================================
SEL_MAX_MISSING = 0.20          # máx. proporción de faltantes para entrar
SEL_FLAG_PREVALENCE = (0.02, 0.98)  # flags binarios: prevalencia mínima/máxima
SEL_CORR_THRESHOLD = 0.95       # |rho Spearman| arriba del cual se agrupan
SEL_MIN_EFFECT = 0.05           # efecto univariado mínimo (|r| o V de Cramér)
# Tope del panel final: si sobreviven más features al embudo, se recorta a este
# número usando un ranking combinado (importancia + efecto univariado).
# None = sin límite | 20 = recomendado (top-20 no pierde AUC vs panel completo)
# 15 = análisis de sensibilidad (panel ultracompacto)
SEL_MAX_FEATURES = 20

# Variables que SIEMPRE se conservan si pasan los filtros básicos (núcleo clínico)
SEL_CORE_KEEP = [
    "AGE", "gender", "HBA1C", "GLU",
    "Systolic blood pressure", "Diastolic blood pressure",
    "BMI", "SCR", "BU", "ALB", "HB", "PLT",
]
# Núcleo de biomarcadores candidatos Tier-1 (laboratorio + demografía)
NUMERIC_CANDIDATES = [
    "AGE", "gender", "nationality", "Systolic blood pressure",
    "Diastolic blood pressure", "BMI", "GLU", "HBA1C", "TG", "TC",
    "HDL_C", "LDL_C", "BU", "SCR", "SUA", "HB", "PCV", "PLT", "ALT",
    "AST", "GGT", "ALP", "TP", "ALB", "GLO", "TBILI", "DBILI", "LDH_L",
    "PT", "APTT", "PTA", "FBG",
]

# ============================================================================
# 5. MODELOS (paso 03)
# ============================================================================
MODELS_ENABLED = [
    "logreg", "random_forest", "xgboost", "lightgbm", "catboost", "mlp",
]
# Los que requieren imputación + escalado (los árboles manejan NaN nativos)
NEEDS_IMPUTE_SCALE = ["logreg", "mlp"]

USE_GPU = True                  # intenta GPU (RTX 5070); si falla -> CPU
GPU_FALLBACK = True             # permite el fallback automático

# Grillas de hiperparámetros (tamaño moderado: tu máquina las barre rápido)
GRIDS = {
    "logreg": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    },
    "random_forest": {
        "clf__n_estimators": [400],
        "clf__max_depth": [None, 12],
        "clf__min_samples_leaf": [2, 5],
    },
    "xgboost": {
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.05, 0.1],
        "clf__n_estimators": [200, 400],
        "clf__min_child_weight": [1, 3, 7],
        "clf__subsample": [0.8],
        "clf__colsample_bytree": [0.8],
    },
    "lightgbm": {
        "clf__num_leaves": [31, 63],
        "clf__learning_rate": [0.05, 0.1],
        "clf__n_estimators": [300, 600],
        "clf__min_child_samples": [20, 50],
        "clf__subsample": [0.8],
        "clf__colsample_bytree": [0.8],
    },
    "catboost": {
        "clf__iterations": [300, 600],
        "clf__learning_rate": [0.05, 0.1],
        "clf__depth": [4, 6],
        "clf__l2_leaf_reg": [1.0, 5.0],
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(128, 64), (256, 128)],
        "clf__alpha": [0.001, 0.01],
        "clf__learning_rate_init": [0.001],
    },
}

# Modo rápido para iterar (RD_QUICK=1 python ...): grillas chicas + menos folds
QUICK_MODE = os.environ.get("RD_QUICK", "0") == "1"
if QUICK_MODE:
    CV_FOLDS = 3
    GRIDS = {
        "logreg": {"clf__C": [0.1, 1.0]},
        "random_forest": {"clf__n_estimators": [300], "clf__max_depth": [None]},
        "xgboost": {"clf__max_depth": [3, 6], "clf__learning_rate": [0.1],
                    "clf__n_estimators": [200], "clf__min_child_weight": [3]},
        "lightgbm": {"clf__num_leaves": [31], "clf__learning_rate": [0.1],
                     "clf__n_estimators": [300], "clf__min_child_samples": [20]},
        "catboost": {"clf__iterations": [300], "clf__learning_rate": [0.1],
                     "clf__depth": [6]},
        "mlp": {"clf__hidden_layer_sizes": [(128, 64)], "clf__alpha": [0.001]},
    }
    ABLATION_KS = (1, 3, 5, 10)

# ============================================================================
# 6. EVALUACIÓN (paso 04)
# ============================================================================
BOOTSTRAP_N = 1000              # remuestreos para IC del AUC

# Ablación por número de features (dentro del paso 04): valores de K a evaluar
# y los porcentajes del AUC máximo que definen el punto de saturación.
ABLATION_KS = (1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30)
ABLATION_TARGETS = (0.99, 0.995)
SINGLE_FEATURE_BASELINE = "HBA1C"   # se anota cuando el top-1 del ranking es esta
SHAP_MAX_SAMPLES = 1500         # submuestra para SHAP (RAM)

# Subgrupos para análisis de equidad: nombre -> (columna, operador, valor)
SUBGROUPS = {
    "edad < 60":   ("AGE", "lt", 60),
    "edad >= 60":  ("AGE", "ge", 60),
    "sexo = 0":    ("gender", "eq", 0),
    "sexo = 1":    ("gender", "eq", 1),
    "hipertenso":  ("HYPERTENTION", "eq", 1),
    "no hipertenso": ("HYPERTENTION", "eq", 0),
}
SUBGROUP_MIN_N = 60             # mínimo de pacientes para reportar un subgrupo

# ============================================================================
# 7. VALIDACIÓN EXTERNA (paso 05) — NHANES 2005-2008
# ============================================================================
NHANES_CYCLES = {
    "2005-2006": "D",
    "2007-2008": "E",
}
# Archivos candidatos que contienen la graduación de retinopatía (el primero
# que exista se usa; si ninguno, el script explica cómo descargarlo a mano)
NHANES_DR_FILES = ["ROPD_{sfx}.XPT", "ROP_{sfx}.XPT", "OPXD_{sfx}.XPT"]
NHANES_DR_COLUMNS = ["DRGRADMP", "DRGRADE", "RETGRADE", "DRLEVEL", "DRQDR"]
NHANES_MIN_AGE = 40
NHANES_CSV_NOTE = (
    "También puedes pasar tu propia cohorte externa en CSV con --csv archivo.csv "
    "(debe tener la columna objetivo y nombres de variables coincidentes)."
)
