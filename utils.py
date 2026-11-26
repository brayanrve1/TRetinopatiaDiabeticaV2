# -*- coding: utf-8 -*-
"""
Funciones compartidas del pipeline: logging, carga, modelos con GPU/fallback,
métricas (DeLong, bootstrap, ECE, decision curve) y utilidades estadísticas.
"""
import io
import os
import sys
import json
import warnings
import importlib
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import config as C

# Silencia el warning cosmético de xgboost GPU ("Falling back to prediction
# using DMatrix due to mismatched devices") durante permutation importance.
warnings.filterwarnings("ignore", message=".*Falling back to prediction.*")
warnings.filterwarnings("ignore", message=".*Device is changed from GPU.*")
warnings.filterwarnings("ignore", message=".*No visible GPU.*")


# ============================================================================
# LOGGING
# ============================================================================
class StepLogger:
    """Imprime a consola y guarda todo en outputs/logs/<step>.txt"""

    def __init__(self, step_name: str):
        C.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = C.LOG_DIR / f"{step_name}.txt"
        self.buf = io.StringIO()
        self._write(f"=== {step_name} — {datetime.now().isoformat(timespec='seconds')} ===")

    def _write(self, s):
        print(s)
        self.buf.write(s + "\n")

    def banner(self, t):
        self._write("\n" + "=" * 78 + f"\n  {t}\n" + "=" * 78)

    def info(self, s=""):
        self._write(str(s))

    def close(self):
        self.buf.flush()
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(self.buf.getvalue())
        print(f"\n[ok] Log guardado: {self.path}")


def ensure_dirs():
    for d in (C.CLEAN_DIR, C.FEAT_DIR, C.MODEL_DIR, C.EVAL_DIR, C.EXT_DIR,
              C.LOG_DIR, C.FIG_DIR, C.DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CARGA DE DATOS
# ============================================================================
def load_raw(log=None):
    """Carga el dataset crudo (parsed CSV de la auditoría o el .xlsx)."""
    if C.PARSED_CSV.exists():
        df = pd.read_csv(C.PARSED_CSV)
        if log:
            log.info(f"[i] Datos desde CSV de auditoría: {C.PARSED_CSV} {df.shape}")
        return df
    if not C.RAW_XLSX.exists():
        raise FileNotFoundError(
            f"No existe {C.RAW_XLSX}. Copia tu .xlsx a {C.DATA_DIR}/ "
            f"o genera {C.PARSED_CSV} con el script de auditoría.")
    xl = pd.ExcelFile(C.RAW_XLSX)
    sheet = next((s for s in xl.sheet_names if s.strip().lower() == "summary"),
                 xl.sheet_names[0])
    df = pd.read_excel(xl, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.strip().replace(
                {"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})
            num = pd.to_numeric(s, errors="coerce")
            df[c] = num if num.notna().sum() >= 0.5 * s.notna().sum() else s
    if log:
        log.info(f"[i] Datos parseados de {C.RAW_XLSX} sheet '{sheet}' {df.shape}")
    return df


def load_cleaned(log=None):
    path = C.CLEAN_DIR / "cleaned_data.csv"
    if not path.exists():
        raise FileNotFoundError("Falta cleaned_data.csv -> ejecuta 01_clean_data.py")
    df = pd.read_csv(path)
    if log:
        log.info(f"[i] Datos limpios cargados: {df.shape}")
    return df


def get_split_indices(df, y):
    """Split estratificado 80/20 determinista (mismo seed => mismos índices
    en todos los pasos del pipeline)."""
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=C.TEST_SIZE,
                                 random_state=C.SEED)
    tr, te = next(sss.split(np.zeros(len(y)), y))
    return tr, te


# ============================================================================
# MODELOS (con GPU y fallback a CPU)
# ============================================================================
def available_backends():
    out = {}
    for name, mod in [("xgboost", "xgboost"), ("lightgbm", "lightgbm"),
                      ("catboost", "catboost"), ("shap", "shap")]:
        try:
            m = importlib.import_module(mod)
            out[name] = getattr(m, "__version__", "?")
        except ImportError:
            out[name] = None
    return out


def make_model(name, gpu=False, **overrides):
    """Construye el estimador base de cada familia.
    gpu=True intenta usar la RTX; si el backend no lo soporta, el llamador
    debe reintentar con gpu=False (ver fit_gs_or_fallback)."""
    rs = C.SEED
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=rs)
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(random_state=rs, n_jobs=C.N_JOBS)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        est = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                            tree_method="hist", random_state=rs, n_jobs=C.N_JOBS,
                            device="cuda" if gpu else "cpu")
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(random_state=rs, n_jobs=C.N_JOBS, verbose=-1,
                             device="gpu" if gpu else "cpu")
    elif name == "catboost":
        from catboost import CatBoostClassifier
        est = CatBoostClassifier(random_seed=rs, verbose=0, allow_writing_files=False,
                                 task_type="GPU" if gpu else "CPU")
    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier
        est = MLPClassifier(max_iter=600, early_stopping=True, random_state=rs)
    else:
        raise ValueError(f"Modelo desconocido: {name}")
    est.set_params(**overrides)
    return est


def build_pipeline(name, est):
    """Envuelve el estimador en Pipeline cuando necesita imputación+escalado."""
    if name in C.NEEDS_IMPUTE_SCALE:
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", est)])
    from sklearn.pipeline import Pipeline
    return Pipeline([("clf", est)])


_GPU_PROBE_CACHE = {}


def _nvidia_smi_ok():
    """¿Existe una GPU NVIDIA visible? (evita el fallback silencioso de xgboost)"""
    import subprocess
    import shutil
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=5)
        return out.returncode == 0 and "GPU" in out.stdout
    except Exception:
        return False


def probe_gpu(name, log=None):
    """Prueba rápida de GPU con un fit diminuto; memoizada por backend."""
    if name in _GPU_PROBE_CACHE:
        return _GPU_PROBE_CACHE[name]
    ok = False
    if not _nvidia_smi_ok():
        if log:
            log.info(f"[i] {name}: sin GPU NVIDIA visible -> CPU")
    else:
        try:
            rng = np.random.default_rng(0)
            X = rng.normal(size=(200, 4))
            y = (X[:, 0] > 0).astype(int)
            est = make_model(name, gpu=True, **_quick_params(name))
            est.fit(X, y)
            ok = True
            if log:
                log.info(f"[i] {name}: GPU operativa (RTX detectada)")
        except Exception as e:  # noqa: BLE001
            if log:
                log.info(f"[i] {name}: GPU falló ({type(e).__name__}) -> CPU")
    _GPU_PROBE_CACHE[name] = ok
    return ok


def _quick_params(name):
    """Parámetros mínimos para la prueba de GPU (rápida)."""
    if name == "xgboost":
        return dict(n_estimators=10, max_depth=2)
    if name == "lightgbm":
        return dict(n_estimators=10, num_leaves=8)
    if name == "catboost":
        return dict(iterations=10, depth=2)
    return {}


def fit_gs_or_fallback(name, Xtr, ytr, log):
    """GridSearchCV con GPU (si el probe pasa) y paralelización sensata.
    CPU: cada fit mono-hilo y GridSearchCV paraleliza combos (evita thrashing).
    GPU: el modelo usa la GPU y el GridSearch corre secuencial."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    grid = {k: v for k, v in C.GRIDS[name].items()}
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)

    gpu = False
    if C.USE_GPU and C.GPU_FALLBACK and name in ("xgboost", "lightgbm", "catboost"):
        gpu = probe_gpu(name, log)

    try:
        est = make_model(name, gpu=gpu)
        if not gpu and name in ("xgboost", "lightgbm", "random_forest"):
            est.set_params(n_jobs=1)          # cada fit mono-hilo
        if gpu and name != "catboost":
            est.set_params(n_jobs=4)          # hilos de soporte del host
        pipe = build_pipeline(name, est)
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc",
                          n_jobs=1 if gpu else C.N_JOBS, refit=True)
        gs.fit(Xtr, ytr)
        device = "GPU" if gpu else "CPU"
        log.info(f"[i] {name}: entrenado en {device} | "
                 f"AUC CV-interna = {gs.best_score_:.4f}")
        return gs, device
    except Exception as e:  # noqa: BLE001
        if gpu:
            log.info(f"[!] {name}: falló con GPU -> reintento en CPU")
            est = make_model(name, gpu=False)
            if name in ("xgboost", "lightgbm", "random_forest"):
                est.set_params(n_jobs=1)
            pipe = build_pipeline(name, est)
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc",
                              n_jobs=C.N_JOBS, refit=True)
            gs.fit(Xtr, ytr)
            log.info(f"[i] {name}: entrenado en CPU | "
                     f"AUC CV-interna = {gs.best_score_:.4f}")
            return gs, "CPU"
        raise RuntimeError(f"{name} no pudo entrenarse: {e}")


# ============================================================================
# ESTADÍSTICA
# ============================================================================
def cramers_v(x, y):
    try:
        cm = pd.crosstab(x, y)
        if cm.shape[0] < 2 or cm.shape[1] < 2:
            return np.nan
        chi2 = stats.chi2_contingency(cm)[0]
        n = cm.values.sum()
        return float(np.sqrt(chi2 / (n * (min(cm.shape) - 1))))
    except Exception:
        return np.nan


def rank_biserial(a, b):
    """Effect size r del test de Mann-Whitney."""
    try:
        u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
        return float(1 - (2 * u) / (len(a) * len(b)))
    except Exception:
        return np.nan


# ---------------- DeLong ----------------
def _delong_components(preds_sorted, m):
    pos = preds_sorted[:m]
    neg = preds_sorted[m:]
    V10 = ((neg[None, :] < pos[:, None]).mean(axis=1)
           + 0.5 * (neg[None, :] == pos[:, None]).mean(axis=1))
    V01 = ((pos[:, None] > neg[None, :]).mean(axis=0)
           + 0.5 * (pos[:, None] == neg[None, :]).mean(axis=0))
    return V10, V01


def delong_test(y_true, p1, p2):
    """Test de DeLong: compara dos AUCs pareados. Devuelve (z, p_valor)."""
    y = np.asarray(y_true)
    m = int(y.sum())
    n = len(y) - m
    order = np.argsort(-y)
    ys = y[order]
    assert ys[:m].sum() == m
    V10a, V01a = _delong_components(p1[order], m)
    V10b, V01b = _delong_components(p2[order], m)
    S10 = np.cov(np.vstack([V10a, V10b]))
    S01 = np.cov(np.vstack([V01a, V01b]))
    S = np.atleast_2d(S10) / m + np.atleast_2d(S01) / n
    diff = V10a.mean() - V10b.mean()
    var = float(np.array([1, -1]) @ S @ np.array([1, -1]))
    if var <= 0:
        return 0.0, 1.0
    z = diff / np.sqrt(var)
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def bootstrap_auc_ci(y_true, p, n_boot=None, seed=None):
    from sklearn.metrics import roc_auc_score
    n_boot = n_boot or C.BOOTSTRAP_N
    rng = np.random.default_rng(seed if seed is not None else C.SEED)
    y = np.asarray(y_true)
    boots = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        boots.append(roc_auc_score(y[ix], p[ix]))
    return np.percentile(boots, [2.5, 97.5])


def expected_calibration_error(y_true, p, n_bins=10):
    y = np.asarray(y_true, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, total = 0.0, len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += m.sum() / total * abs(y[m].mean() - p[m].mean())
    return float(ece)


def net_benefit(y_true, p, thresholds=None):
    """Decision curve: beneficio neto por umbral."""
    y = np.asarray(y_true, dtype=float)
    thresholds = thresholds if thresholds is not None else np.linspace(0.01, 0.99, 199)
    prev = y.mean()
    nb_model, nb_all = [], []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum() / len(y)
        fp = ((pred == 1) & (y == 0)).sum() / len(y)
        nb_model.append(tp - fp * t / (1 - t))
        nb_all.append(prev - (1 - prev) * t / (1 - t))
    return np.asarray(thresholds), np.asarray(nb_model), np.asarray(nb_all)


def correct_prevalence(p, prev_sample, prev_real):
    """Ajuste bayesiano de probabilidades caso-control -> prevalencia real."""
    k = (prev_real / (1 - prev_real)) / (prev_sample / (1 - prev_sample))
    odds = np.asarray(p) / np.clip(1 - np.asarray(p), 1e-12, None)
    o2 = odds * k
    return o2 / (1 + o2)


def triage_metrics(y_true, p, sens_targets=(0.85, 0.90, 0.95),
                   prev_real=None):
    """Tabla de puntos operativos a sensibilidad fija + métricas a prevalencia real."""
    from sklearn.metrics import roc_curve
    prev_real = prev_real if prev_real is not None else C.REAL_PREVALENCE
    y = np.asarray(y_true)
    fpr, tpr, thr = roc_curve(y, p)
    rows = []
    for ts in sens_targets:
        i = np.where(tpr >= ts)[0]
        if len(i) == 0:
            continue
        j = i[0]
        t = float(thr[j])
        sens, spec = float(tpr[j]), float(1 - fpr[j])
        ppv = sens * prev_real / max(sens * prev_real + (1 - spec) * (1 - prev_real), 1e-12)
        npv = spec * (1 - prev_real) / max(spec * (1 - prev_real) + (1 - sens) * prev_real, 1e-12)
        rows.append(dict(sens_obj=ts, umbral=round(t, 4),
                         sensibilidad=round(sens, 3), especificidad=round(spec, 3),
                         tasa_derivacion=round(float((p >= t).mean()), 3),
                         PPV_prev_real=round(ppv, 3), NPV_prev_real=round(npv, 3)))
    return pd.DataFrame(rows)


def deployment_cohort_metrics(y_true, p, threshold, prev_real=None, seed=None):
    """Simula una cohorte de despliegue con prevalencia real (remuestreo)."""
    prev_real = prev_real if prev_real is not None else C.REAL_PREVALENCE
    y = np.asarray(y_true)
    rng = np.random.default_rng(seed if seed is not None else C.SEED)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    N = len(y)
    n_pos = int(round(prev_real * N))
    ix = np.concatenate([rng.choice(pos, n_pos, replace=True),
                         rng.choice(neg, N - n_pos, replace=True)])
    yy, pp = y[ix], p[ix]
    pred = (pp >= threshold).astype(int)
    tn, fp, fn, tp = [int(x) for x in np.bincount(
        np.arange(4), yy * 2 + pred, minlength=4)] if False else [
        int(((pred == 0) & (yy == 0)).sum()), int(((pred == 1) & (yy == 0)).sum()),
        int(((pred == 0) & (yy == 1)).sum()), int(((pred == 1) & (yy == 1)).sum())]
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return dict(prevalencia=round(yy.mean(), 3), umbral=round(float(threshold), 4),
                sensibilidad=round(sens, 3), especificidad=round(spec, 3),
                PPV=round(ppv, 3), NPV=round(npv, 3),
                tasa_derivacion=round(float(pred.mean()), 3))


def hard_exit():
    """Terminación dura (os._exit) DESPUÉS de guardar todos los artefactos.
    Evita los crashes de cierre (exit code 0xC0000409 en Windows) producidos
    por el teardown de DLLs nativas de xgboost/lightgbm/catboost/shap."""
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(0)


def save_json(obj, path):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
