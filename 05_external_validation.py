# -*- coding: utf-8 -*-
"""
PASO 05 — VALIDACIÓN EXTERNA CON NHANES (ciclos según config.NHANES_CYCLES)
============================================================================
Evalúa las DOS especificaciones finales del algoritmo ganador del benchmark
en una cohorte externa independiente: adultos >= 40 años con diabetes y fondo
de ojo graduable (archivo OPXRET, variable OPDURET del peor ojo; niveles
10-13 = sin retinopatía, >= 14 = retinopatía diabética presente).

Uso:
    python 05_external_validation.py
    python 05_external_validation.py --sin-descarga     # solo usa el caché
    python 05_external_validation.py --solo-cohorte     # sin evaluación

Salidas: outputs/external/* (tablas, figuras, external_summary.json y log).
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             brier_score_loss)

import config as C
import utils as U

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ---------------------------------------------------------------------------
# Decisiones de mapeo (verificadas en los codebooks oficiales del CDC)
# ---------------------------------------------------------------------------
GENDER_FEMALE = 0
# Archivos XPT por ciclo (OPXRET = graduación de retinopatía; GHB = HbA1c;
# ojo: en 2005-2008 la HbA1c se llama GHB, desde 2009 pasa a GH)
NHANES_FILES = ["DEMO", "BMX", "BPX", "BIOPRO", "CBC", "GHB", "GLU",
                "DIQ", "MCQ", "OPXRET"]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
MIN_AGE = C.NHANES_MIN_AGE
# Graduación: niveles 10-13 = sin RD (12 = otra enfermedad retinal,
# 13 = dudosa); >= 14 = retinopatía diabética presente.
DR_LEVEL_MIN = 14
# Features que NHANES no puede aportar ni aproximar: se imputan del train.
NO_NATIVAS = ["PT", "LEADDP"]
# Features binarias del panel (se rellenan con moda, no mediana)
BINARIAS = {"LEADDP", "CHD", "ENDOCRINE_DISEASE", "gender"}

FIG_DIR = C.EXT_DIR / "figs"


def url_candidates(period, sfx, stem):
    """URLs candidatas en orden de preferencia. La oficial (la que enlaza la
    página de datos del CDC, verificada) es la primera:
    https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{anio}/DataFiles/{FILE}_{sfx}.xpt
    donde {anio} es SOLO el año inicial del ciclo (2005, 2007)."""
    year = period.split("-")[0]
    base = ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"
            f"{year}/DataFiles/{stem}_{sfx}")
    return [
        f"{base}.xpt",
        f"{base}.XPT",
        f"https://wwwn.cdc.gov/nchs/nhanes/{period.lower()}/{stem}_{sfx}.XPT",
        f"https://wwwn.cdc.gov/nchs/data/nhanes/{period.lower()}/{stem}_{sfx}.XPT",
    ]


def _cache_path(cache_dir, stem, sfx):
    """Busca el archivo en caché con cualquier capitalización de extensión."""
    for ext in (".xpt", ".XPT", ".Xpt"):
        p = cache_dir / f"{stem}_{sfx}{ext}"
        if p.exists():
            return p
    return cache_dir / f"{stem}_{sfx}.xpt"


def _es_xpt_valido(path):
    """Un XPT real pesa > 100 KB y empieza con la cabecera SAS 'HEADER RECORD';
    el CDN del CDC a veces responde 200 con una página HTML de error."""
    try:
        if path.stat().st_size < 100_000:
            return False
        with open(path, "rb") as f:
            head = f.read(32)
        return head.startswith(b"HEADER")
    except OSError:
        return False


def download_xpt(period, sfx, stem, cache_dir, log, skip=False):
    dest = _cache_path(cache_dir, stem, sfx)
    if dest.exists() and _es_xpt_valido(dest):
        log.info(f"   [cache] {dest.name} válido ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    if dest.exists():
        log.info(f"   [!] {dest.name} en caché no es un XPT válido -> se re-descarga")
        dest.unlink()
    if skip:
        log.info(f"   [!] {dest.name} no está en caché y --sin-descarga activo")
        return None
    for url in url_candidates(period, sfx, stem):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=240) as r, \
                    open(dest, "wb") as f:
                f.write(r.read())
        except Exception as e:  # noqa: BLE001
            log.info(f"   [i] fallo {url}: {e}")
            continue
        if _es_xpt_valido(dest):
            log.info(f"   [ok] {dest.name} descargado "
                     f"({dest.stat().st_size/1e6:.1f} MB)")
            return dest
        log.info(f"   [!] {url} devolvió una página, no un XPT -> otro patrón")
        dest.unlink()
    log.info(f"   [!] {stem}_{sfx}: ninguna URL funcionó. Descárgalo a mano de "
             f"https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx , guárdalo "
             f"como {dest} y relanza con --sin-descarga.")
    return None


def pick(df, *names):
    """Primera columna existente, o serie de NaN."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(np.nan, index=df.index)


def build_cycle(period, sfx, cache_dir, log, skip):
    """Descarga/fusiona un ciclo y lo armoniza. Devuelve None si el ciclo no
    tiene sus archivos imprescindibles (se omite en vez de abortar)."""
    log.banner(f"CICLO {period} (sufijo {sfx})")
    parts = {}
    for stem in NHANES_FILES:
        p = download_xpt(period, sfx, stem, cache_dir, log, skip)
        if p is None:
            parts[stem] = None
            continue
        try:
            df = pd.read_sas(p)
            df.columns = [str(c).upper() for c in df.columns]
            parts[stem] = df
            log.info(f"   [i] {stem}: {df.shape}")
        except Exception as e:  # noqa: BLE001
            log.info(f"   [!] no se pudo leer {p.name}: {e}")
            parts[stem] = None

    if parts.get("DEMO") is None or parts.get("OPXRET") is None:
        log.info(f"   [!] Ciclo {period} OMITIDO: faltan DEMO u OPXRET.")
        log.info("       Para incluirlo, descarga sus XPT a la carpeta de caché "
                 "o lanza sin --sin-descarga para bajarlos automáticamente.")
        return None

    merged = parts["DEMO"]
    for stem in NHANES_FILES[1:]:
        if parts.get(stem) is not None:
            merged = merged.merge(parts[stem], on="SEQN", how="left")
    return harmonize(merged, period, log)


# ---------------------------------------------------------------------------
# B-C. ARMONIZACIÓN A LAS UNIDADES DEL DATASET LIMPIO (paso 01)
# ---------------------------------------------------------------------------
def harmonize(g, period, log):
    out = pd.DataFrame({"SEQN": g["SEQN"], "ciclo": period})

    # Demografía -------------------------------------------------------------
    out["AGE"] = pick(g, "RIDAGEYR")
    sexo = pick(g, "RIAGENDR")              # 1 = varón, 2 = mujer
    out["gender"] = sexo.map({1: 1 - GENDER_FEMALE, 2: GENDER_FEMALE})
    if "RIDRETH1" in g.columns:
        out["RIDRETH1"] = g["RIDRETH1"]

    # Presión arterial: promedio de las lecturas disponibles ------------------
    # En 2005-2008 la diastólica se llama BPXDI1..4 (BPXDIA* es posterior)
    sys_cols = [c for c in ("BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4") if c in g.columns]
    dia_cols = [c for c in ("BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4",
                            "BPXDIA1", "BPXDIA2", "BPXDIA3", "BPXDIA4")
                if c in g.columns]
    out["Systolic blood pressure"] = g[sys_cols].mean(axis=1) if sys_cols else np.nan
    out["Diastolic blood pressure"] = g[dia_cols].mean(axis=1) if dia_cols else np.nan
    out["BMI"] = pick(g, "BMXBMI")

    # Laboratorio (conversiones hacia las unidades del dataset limpio) --------
    glu_si, glu_mg = pick(g, "LBDGLUSI"), pick(g, "LBXGLU")
    out["GLU"] = glu_si.where(glu_si.notna(), glu_mg / 18.0)          # mmol/L
    out["HBA1C"] = pick(g, "LBXGH")                                    # %
    bu_si, bu_mg = pick(g, "LBDSBUSI"), pick(g, "LBXSBU")
    out["BU"] = bu_si.where(bu_si.notna(), bu_mg / 2.8)                # mmol/L urea
    scr_si, scr_mg = pick(g, "LBDSCRSI"), pick(g, "LBXSCR")
    out["SCR"] = scr_si.where(scr_si.notna(), scr_mg * 88.4)           # umol/L
    alb_si, alb = pick(g, "LBDSALSI"), pick(g, "LBXSAL")
    out["ALB"] = alb_si.where(alb_si.notna(), alb * 10.0)              # g/L
    tp_si, tp = pick(g, "LBDSTPSI"), pick(g, "LBXSTP")
    out["TP"] = tp_si.where(tp_si.notna(), tp * 10.0)                  # g/L
    glo_si, glo = pick(g, "LBDSGBSI"), pick(g, "LBXSGB")
    out["GLO"] = glo_si.where(glo_si.notna(), glo * 10.0)              # g/L
    tb_si, tb = pick(g, "LBDSTBSI"), pick(g, "LBXSTB")
    out["TBILI"] = tb_si.where(tb_si.notna(), tb * 17.1)               # umol/L
    db_si, db = pick(g, "LBDSDBSI"), pick(g, "LBXSDB")
    out["DBILI"] = db_si.where(db_si.notna(), db * 17.1)               # umol/L
    out["GGT"] = pick(g, "LBXSGTSI", "LBXSGT")                         # U/L
    # Hemograma 2005-2008: LBXHCT (hematocrito %), LBXPLTSI (plaquetas),
    # LBXHGB (hemoglobina g/dL). LBXPCV/LBXPLT/LBXHB no existen en estos ciclos.
    out["PCV"] = pick(g, "LBXHCT", "LBXPCV")    # %: el paso 01 deja PCV en %
    out["PLT"] = pick(g, "LBXPLTSI", "LBXPLT")  # 10^9/L == 10^3/uL
    out["HB"] = pick(g, "LBXHGB", "LBXHB") * 10.0  # g/L (fuera del panel)
    out["PT"] = np.nan                  # NHANES no mide coagulación

    # Proxies de flags clínicos ------------------------------------------------
    # Las letras de MCQ160 cambian por ciclo. En 2005-2008 (verificado en los
    # codebooks): B=ICC, C=cardiopatía coronaria, D=angina, E=infarto,
    # F=ictus, M=tiroides (en 2011+ la tiroides pasa a MCQ160N).
    chd_items = [(g[c] == 1) for c in ("MCQ160C", "MCQ160D", "MCQ160E")
                 if c in g.columns]
    out["CHD"] = (pd.concat(chd_items, axis=1).max(axis=1).astype(float)
                  if chd_items else np.nan)
    endo = None
    for c in ("MCQ160M", "MCQ170M", "MCQ160N"):
        if c in g.columns:
            endo = (g[c] == 1).astype(float)
            break
    out["ENDOCRINE_DISEASE"] = endo if endo is not None else np.nan
    out["LEADDP"] = np.nan              # flag de duración: sin equivalente directo

    # Duración de diabetes (informativa; el flag LEADDP no se reconstruye) ----
    dx_age = pick(g, "DID040", "DIQ040", "DID040G")
    dur = out["AGE"] - dx_age
    out["DIABETES_DURATION"] = dur.where((dur >= 0) & (dur <= 90))

    # Diabetes (criterio compuesto) -------------------------------------------
    diq_dx = pick(g, "DIQ010", "DID010")
    out["diabetes_autorreporte"] = (diq_dx == 1).astype(float)
    out["es_diabetico"] = ((diq_dx == 1)
                           | (out["HBA1C"] >= 6.5)
                           | (out["GLU"] >= 7.0)).astype(float)

    # Objetivo: graduación de retinopatía (peor ojo) ---------------------------
    lev = pick(g, "OPDURET")
    if lev.isna().all():
        dd, ds = pick(g, "OPDDRET"), pick(g, "OPDSRET")
        lev = pd.concat([dd, ds], axis=1).max(axis=1)
    out["DR_LEVEL"] = lev
    out["foto_graduable"] = lev.notna().astype(float)
    out["RD"] = (lev >= DR_LEVEL_MIN).astype(float)
    out.loc[lev.isna(), "RD"] = np.nan

    log.info(f"   [i] armonizado {period}: {out.shape}")
    return out


def prepare_cohort(cache_dir, log, skip_download):
    frames, usados = [], []
    for period, sfx in C.NHANES_CYCLES.items():
        fr = build_cycle(period, sfx, cache_dir, log, skip_download)
        if fr is not None:
            frames.append(fr)
            usados.append(period)
    if not frames:
        raise RuntimeError("Ningún ciclo disponible: revisa la caché de XPT o "
                           "lanza sin --sin-descarga.")
    coh = pd.concat(frames, ignore_index=True)

    log.banner("COHORTE ANALÍTICA")
    log.info(f"   ciclos procesados: {usados}")
    n0 = len(coh)
    coh = coh[coh["AGE"] >= MIN_AGE]
    log.info(f"   adultos >= {MIN_AGE} años: {n0} -> {len(coh)}")
    coh = coh[coh["foto_graduable"] == 1]
    log.info(f"   con foto de fondo de ojo graduable: {len(coh)}")
    diab = coh[coh["es_diabetico"] == 1].copy()
    log.info(f"   con diabetes (autorreporte, HbA1c>=6.5 o GLU>=7): {len(diab)}")
    diab = diab[diab["RD"].notna()].copy()
    prev = float(diab["RD"].mean())
    log.info(f"   cohorte final con objetivo válido: {len(diab)} | "
             f"prevalencia de RD: {prev:.1%} "
             f"(config.REAL_PREVALENCE = {C.REAL_PREVALENCE:.0%})")
    if len(diab) < 200:
        log.info("   [!] cohorte pequeña: agrega el ciclo 2007-2008 si es posible")
    return coh, diab, usados


# ---------------------------------------------------------------------------
# D. EVALUACIÓN EXTERNA (dos especificaciones del algoritmo ganador)
# ---------------------------------------------------------------------------
def imputation_plan(feats, diab, dft, log):
    """Política de cada feature del panel en NHANES y relleno de faltantes.
    Devuelve también el conjunto de features 100 % imputadas (sin dato real en
    NHANES): esas son las únicas que se excluyen de la versión transportable."""
    rows = []
    fills = {}
    fully_imputed = set()
    for f in feats:
        col = diab[f] if f in diab.columns else pd.Series(np.nan, index=diab.index)
        frac_nan = float(col.isna().mean())
        tr = dft[f].dropna()
        fill = None
        if f in NO_NATIVAS or frac_nan >= 0.999:
            fill = (float(tr.mode().iloc[0]) if f in BINARIAS
                    else float(tr.median())) if len(tr) else 0.0
            policy = "imputada_moda" if f in BINARIAS else "imputada_mediana"
        elif frac_nan > 0 and f in ("CHD", "ENDOCRINE_DISEASE"):
            fill = float(tr.mode().iloc[0]) if len(tr) else 0.0
            policy = "proxy (faltantes -> moda train)"
        elif frac_nan > 0:
            fill = float(tr.median()) if len(tr) else 0.0
            policy = "nativa (faltantes -> mediana train)"
        else:
            policy = "nativa"
        if frac_nan >= 0.99:
            fully_imputed.add(f)
        if fill is not None:
            fills[f] = fill
        rows.append(dict(feature=f, politica=policy,
                         pct_faltante_nhanes=round(100 * frac_nan, 2),
                         valor_relleno="" if fill is None else round(fill, 4)))
        if f in fills:
            log.info(f"   [relleno] {f}: {policy} -> {fills[f]:.4g} (del train)")
    return pd.DataFrame(rows), fills, fully_imputed


def evaluate_model(name, etiqueta, pipe, X, y, log):
    p = pipe.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p)
    lo, hi = U.bootstrap_auc_ci(y, p)
    row = dict(modelo=name, cohorte=etiqueta, n=int(len(y)),
               prevalencia=round(float(y.mean()), 3),
               auroc=round(auc, 4), ic95=f"[{lo:.4f}, {hi:.4f}]",
               auprc=round(average_precision_score(y, p), 4),
               brier=round(brier_score_loss(y, p), 4),
               ece=round(U.expected_calibration_error(y, p), 4))
    log.info(f"   {name}: AUROC={auc:.4f} IC95[{lo:.4f},{hi:.4f}] "
             f"AUPRC={row['auprc']} Brier={row['brier']} ECE={row['ece']}")
    return row, p


def main():
    U.ensure_dirs()
    C.EXT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(C.DATA_DIR / "nhanes"))
    ap.add_argument("--sin-descarga", action="store_true",
                    help="usa solo los XPT ya descargados")
    ap.add_argument("--solo-cohorte", action="store_true",
                    help="se detiene después de preparar la cohorte")
    args = ap.parse_args()

    log = U.StepLogger("05_external_validation")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- A-C: descarga, fusión y cohorte -----------------------------------
    merged_all, diab, ciclos_usados = prepare_cohort(cache_dir, log,
                                                     args.sin_descarga)
    etiqueta = "NHANES " + " + ".join(ciclos_usados)
    merged_all.to_csv(C.EXT_DIR / "nhanes_merged.csv", index=False)

    if args.solo_cohorte:
        log.info("[ok] --solo-cohorte: cohortes guardadas, se detiene aquí.")
        log.close()
        U.hard_exit()

    # ---- Requisitos del modelo ganador --------------------------------------
    split_p = C.MODEL_DIR / "split.json"
    cv_p = C.MODEL_DIR / "cv_metrics.csv"
    if not split_p.exists() or not cv_p.exists():
        log.banner("PENDIENTE")
        log.info("[!] Aún no hay modelos entrenados (pasos 03/04).")
        log.info(f"    La cohorte quedó en {C.EXT_DIR / 'nhanes_merged.csv'}")
        log.close()
        U.hard_exit()

    split = U.load_json(split_p)
    feats = split["features"]
    df_clean = U.load_cleaned(log)
    y_all = df_clean[C.TARGET].astype(int)
    dft = df_clean.iloc[split["train_idx"]]
    ytr = y_all.iloc[split["train_idx"]]

    cv_tbl = pd.read_csv(cv_p)
    cv_tbl = cv_tbl[cv_tbl["estado"] == "ok"].sort_values("auc_cv", ascending=False)
    best = str(cv_tbl.iloc[0]["modelo"])
    meta = U.load_json(C.MODEL_DIR / f"{best}_meta.json")
    bp = {k.split("__")[-1]: v for k, v in meta["mejores_params"].items()}
    log.banner(f"ALGORITMO GANADOR DEL BENCHMARK: {best}")
    log.info(f"   mejores parámetros: {bp}")

    # ---- C: imputación documentada ------------------------------------------
    log.banner("POLÍTICA DE FEATURES EN NHANES")
    imp_tbl, fills, fully_imputed = imputation_plan(feats, diab, dft, log)
    imp_tbl.to_csv(C.EXT_DIR / "imputation_table.csv", index=False)
    nativas = [f for f in feats if f not in fully_imputed]
    log.info(f"   Versiones a evaluar:")
    log.info(f"     PRINCIPAL     : panel completo ({len(feats)} variables, "
             f"{len(fills)} con algún relleno)")
    log.info(f"     TRANSPORTABLE : solo nativas ({len(nativas)} variables, "
             f"excluye {sorted(fully_imputed)})")

    X_ext = diab[feats].astype(float).copy()
    for f, v in fills.items():
        X_ext[f] = X_ext[f].fillna(v)
    y_ext = diab["RD"].astype(int)

    # Sanity de unidades: mediana train vs mediana NHANES ----------------------
    log.banner("SANITY DE UNIDADES (mediana train vs mediana NHANES)")
    san_rows = []
    for f in feats:
        a = float(dft[f].median())
        b = float(X_ext[f].median())
        ratio = (b / a) if a not in (0,) and pd.notna(a) else np.nan
        flag = "" if (pd.isna(ratio) or 0.5 <= ratio <= 2.0) else "  [REVISAR UNIDAD]"
        san_rows.append(dict(feature=f, mediana_train=round(a, 3),
                             mediana_nhanes=round(b, 3),
                             ratio=None if pd.isna(ratio) else round(ratio, 3)))
        log.info(f"   {f:<26} train={a:>10.3f}  nhanes={b:>10.3f}  "
                 f"ratio={ratio if pd.notna(ratio) else float('nan'):>6.2f}{flag}")
    pd.DataFrame(san_rows).to_csv(C.EXT_DIR / "unit_sanity.csv", index=False)

    # ---- D: evaluación de las dos especificaciones ---------------------------
    nom_principal = f"{best} principal (panel {len(feats)})"
    nom_transp = f"{best} transportable (panel {len(nativas)})"

    log.banner(f"EVALUACIÓN EXTERNA — {etiqueta}")
    pipe20 = joblib.load(C.MODEL_DIR / f"{best}.joblib")
    row_full, p_full = evaluate_model(nom_principal, etiqueta, pipe20,
                                      X_ext, y_ext, log)
    gpu_b = (U.probe_gpu(best, log) if C.USE_GPU and best in
             ("xgboost", "lightgbm", "catboost") else False)
    m_nat = U.build_pipeline(best, U.make_model(best, gpu=gpu_b, **bp))
    m_nat.fit(dft[nativas].astype(float), ytr)
    row_nat, p_nat = evaluate_model(nom_transp, etiqueta, m_nat,
                                    X_ext[nativas], y_ext, log)

    z, pv = U.delong_test(y_ext.values, p_full, p_nat)
    sig = "SIGNIFICATIVA" if pv < 0.05 else "NO significativa"
    log.info(f"\n   DeLong externo principal vs transportable: z={z:.2f} "
             f"p={pv:.4f} -> diferencia {sig}")
    if pv >= 0.05:
        log.info("   La versión transportable es estadísticamente equivalente "
                 "en el exterior y no depende de variables ausentes.")

    # Punto operativo de triaje (ambas versiones) --------------------------------
    log.banner("PUNTO OPERATIVO DE TRIAJE EN NHANES (sensibilidad fija)")
    tri_ext = U.triage_metrics(y_ext.values, p_full)
    tri_ext.insert(0, "modelo", nom_principal)
    tri_nat = U.triage_metrics(y_ext.values, p_nat)
    tri_nat.insert(0, "modelo", nom_transp)
    tri_all = pd.concat([tri_ext, tri_nat])
    tri_all.to_csv(C.EXT_DIR / "triage_external.csv", index=False)
    log.info(tri_all.to_string(index=False))

    thr_interno = None
    tri_p = C.EVAL_DIR / "triage_thresholds.csv"
    if tri_p.exists():
        tri04 = pd.read_csv(tri_p)
        sel = tri04[(tri04["modelo"] == best)
                    & (tri04["sens_obj"] == C.TARGET_SENSITIVITY)]
        if len(sel):
            thr_interno = float(sel.iloc[0]["umbral"])
    if thr_interno is not None:
        log.info(f"\n   Umbral interno del paso 04 (sens>={C.TARGET_SENSITIVITY:.0%} "
                 f"en test): {thr_interno:.4f}")
        yy = y_ext.values
        for lab, p_ in ((nom_principal, p_full), (nom_transp, p_nat)):
            pred = (p_ >= thr_interno).astype(int)
            tp = int(((pred == 1) & (yy == 1)).sum())
            fp = int(((pred == 1) & (yy == 0)).sum())
            fn = int(((pred == 0) & (yy == 1)).sum())
            tn = int(((pred == 0) & (yy == 0)).sum())
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            ppv = tp / max(tp + fp, 1)
            npv = tn / max(tn + fn, 1)
            log.info(f"   {lab}: sens={sens:.3f} spec={spec:.3f} PPV={ppv:.3f} "
                     f"NPV={npv:.3f} derivación={pred.mean():.3f}")

    # Subgrupos de equidad -----------------------------------------------------
    log.banner("SUBGRUPOS DE EQUIDAD EN NHANES (versión principal)")
    sg_rows = []
    subs = {"edad < 60": diab["AGE"] < 60, "edad >= 60": diab["AGE"] >= 60,
            f"gender = {GENDER_FEMALE}": diab["gender"] == GENDER_FEMALE,
            f"gender = {1 - GENDER_FEMALE}": diab["gender"] == (1 - GENDER_FEMALE)}
    if "RIDRETH1" in diab.columns:
        for k, lab in {1: "mex-am", 2: "hispano", 3: "blanco NH",
                       4: "negro NH", 5: "otros"}.items():
            subs[f"raza {lab}"] = diab["RIDRETH1"] == k
    for label, mask in subs.items():
        mask = mask.values.astype(bool)
        if mask.sum() < C.SUBGROUP_MIN_N or len(np.unique(y_ext.values[mask])) < 2:
            continue
        auc = roc_auc_score(y_ext.values[mask], p_full[mask])
        sg_rows.append(dict(subgrupo=label, n=int(mask.sum()), auroc=round(auc, 4)))
        log.info(f"   {label:<18} n={int(mask.sum()):>5}  AUROC={auc:.4f}")
    pd.DataFrame(sg_rows).to_csv(C.EXT_DIR / "subgroup_external.csv", index=False)

    # Figuras -------------------------------------------------------------------
    if HAS_PLOT:
        from sklearn.calibration import calibration_curve
        fpr, tpr, _ = roc_curve(y_ext, p_full)
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.plot(fpr, tpr, color="#2c6e49",
                label=f"principal (AUROC {row_full['auroc']:.3f})")
        fpr2, tpr2, _ = roc_curve(y_ext, p_nat)
        ax.plot(fpr2, tpr2, color="#b5651d", ls="--",
                label=f"transportable (AUROC {row_nat['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("1 - Especificidad"); ax.set_ylabel("Sensibilidad")
        ax.set_title(f"Validación externa {etiqueta} (ROC)")
        ax.legend(fontsize=9); fig.tight_layout()
        fig.savefig(FIG_DIR / "roc_external.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        for p_, lab in ((p_full, "principal (panel completo)"),
                        (p_nat, "transportable")):
            pt, pp = calibration_curve(y_ext, p_, n_bins=8, strategy="quantile")
            ax.plot(pp, pt, "o-", label=lab, ms=4)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Frecuencia observada")
        ax.set_title(f"Calibración externa ({etiqueta})")
        ax.legend(fontsize=9); fig.tight_layout()
        fig.savefig(FIG_DIR / "calibration_external.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        tt = np.linspace(0.01, 0.99, 199)
        for p_, lab in ((p_full, "principal"), (p_nat, "transportable")):
            _, nb, _ = U.net_benefit(y_ext.values, p_, tt)
            ax.plot(tt, nb, label=lab)
        _, _, nb_all = U.net_benefit(y_ext.values, p_full, tt)
        ax.plot(tt, nb_all, "k--", label="derivar a todos")
        ax.axhline(0, color="gray", ls=":")
        ax.set_xlabel("Umbral"); ax.set_ylabel("Beneficio neto")
        ax.set_title(f"Decision curve — {etiqueta} (prevalencia observada)")
        ax.legend(fontsize=9); fig.tight_layout()
        fig.savefig(FIG_DIR / "decision_curve_external.png", dpi=130); plt.close(fig)
        log.info("\n[i] Figuras: roc_external.png, calibration_external.png, "
                 "decision_curve_external.png")

    # Resumen -------------------------------------------------------------------
    diab.to_csv(C.EXT_DIR / "nhanes_cohort.csv", index=False)
    metrics = pd.DataFrame([row_full, row_nat])
    metrics.to_csv(C.EXT_DIR / "external_metrics.csv", index=False)
    U.save_json({
        "etiqueta_cohorte": etiqueta,
        "ciclos_usados": ciclos_usados,
        "cohorte_n": int(len(diab)),
        "prevalencia_rd": round(float(diab["RD"].mean()), 4),
        "metricas": metrics.to_dict(orient="records"),
        "delong_principal_vs_transportable": {"z": round(float(z), 3),
                                             "p": round(float(pv), 4)},
        "umbral_interno_paso04": thr_interno,
        "politicas_features": imp_tbl.to_dict(orient="records"),
    }, C.EXT_DIR / "external_summary.json")

    log.banner("FIN")
    log.info(f"Cohorte: {etiqueta} | n={len(diab)} | prevalencia RD="
             f"{float(diab['RD'].mean()):.1%}")
    log.info(f"Escribe en la tesis: 'validación externa en {etiqueta}, "
             f"n = {len(diab)}'. Resultados en {C.EXT_DIR}")
    log.close()
    U.hard_exit()


if __name__ == "__main__":
    main()
