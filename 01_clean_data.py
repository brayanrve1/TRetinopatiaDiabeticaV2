# -*- coding: utf-8 -*-
"""
PASO 01 — LIMPIEZA DOCUMENTADA
Carga el dataset crudo y aplica correcciones registradas paso a paso:
deduplicación, columnas descartadas y corrección de unidades.
Salidas: outputs/clean/cleaned_data.csv + cleaning_log.json
"""
import numpy as np
import pandas as pd

import config as C
import utils as U


def main():
    U.ensure_dirs()
    log = U.StepLogger("01_clean_data")

    log.banner("CARGA")
    df = U.load_raw(log)

    log.banner("LIMPIEZA DOCUMENTADA")
    report = {"filas_iniciales": int(len(df)), "pasos": []}

    def step(name, n):
        report["pasos"].append(dict(paso=name, contador=int(n)))
        log.info(f"   [clean] {name}: {n}")

    # 1) duplicados
    n0 = len(df)
    if C.ID_COL in df.columns:
        dup = int(df[C.ID_COL].duplicated().sum())
        df = df[~df[C.ID_COL].duplicated(keep="first")].reset_index(drop=True)
        step(f"Case_ID duplicados eliminados (keep first) ({n0} -> {len(df)})", dup)

    # 2) columnas descartadas por auditoría
    present = [c for c in C.DROP_ALWAYS if c in df.columns]
    df = df.drop(columns=present, errors="ignore")
    step(f"Columnas eliminadas: {present}", len(present))

    # 3) correcciones de unidad
    uf = C.UNIT_FIXES
    if "FBG" in df.columns:
        s = df["FBG"]
        n = int((s > uf["fbg_mgdl_threshold"]).sum())
        df.loc[s > uf["fbg_mgdl_threshold"], "FBG"] = (
            df.loc[s > uf["fbg_mgdl_threshold"], "FBG"] / 18.0)
        df.loc[df["FBG"] < uf["fbg_min_valid"], "FBG"] = np.nan
        step(f"FBG: mg/dL -> mmol/L (/{18}) y <{uf['fbg_min_valid']} -> NaN", n)

    if "BU" in df.columns:
        s = df["BU"]
        n = int((s > uf["bu_threshold"]).sum())
        df.loc[s > uf["bu_threshold"], "BU"] = df.loc[s > uf["bu_threshold"], "BU"] / 2.8
        df["BU"] = df["BU"].clip(upper=uf["bu_clip"])
        step(f"BU: mg/dL -> mmol/L (/2.8) + clip {uf['bu_clip']}", n)

    if "PTA" in df.columns:
        s = df["PTA"]
        n = int((s < uf["pta_fraction_max"]).sum())
        df.loc[s < uf["pta_fraction_max"], "PTA"] = (
            df.loc[s < uf["pta_fraction_max"], "PTA"] * 100.0)
        step("PTA: fracción -> % (x100)", n)

    if "PCV" in df.columns and df["PCV"].max() < 2:
        df["PCV"] = df["PCV"] * 100.0
        step("PCV: fracción -> % (x100)", 1)

    if "HEIGHT" in df.columns and "WEIGHT" in df.columns:
        lo_h, hi_h = uf["height_range"]
        lo_w, hi_w = uf["weight_range"]
        bad_h = int(((df["HEIGHT"] < lo_h) | (df["HEIGHT"] > hi_h)).sum())
        bad_w = int(((df["WEIGHT"] < lo_w) | (df["WEIGHT"] > hi_w)).sum())
        df.loc[(df["HEIGHT"] < lo_h) | (df["HEIGHT"] > hi_h), "HEIGHT"] = np.nan
        df.loc[(df["WEIGHT"] < lo_w) | (df["WEIGHT"] > hi_w), "WEIGHT"] = np.nan
        ok = df["HEIGHT"].notna() & df["WEIGHT"].notna()
        df.loc[ok, "BMI"] = (df.loc[ok, "WEIGHT"] / (df.loc[ok, "HEIGHT"] / 100) ** 2).round(3)
        step(f"HEIGHT/WEIGHT imposibles -> NaN y BMI recalculado", bad_h + bad_w)

    if "SCR" in df.columns:
        n = int((df["SCR"] > uf["scr_max"]).sum())
        df.loc[df["SCR"] > uf["scr_max"], "SCR"] = np.nan
        step(f"SCR > {uf['scr_max']} -> NaN", n)

    # 4) sanity del objetivo
    y = df[C.TARGET].astype(int)
    log.banner("OBJETIVO")
    log.info(f"Columna '{C.TARGET}': {y.value_counts().to_dict()}")
    report["prevalencia_muestra"] = float(y.mean())
    if abs(y.mean() - 0.5) < 0.02:
        log.info("   [!] Balance ~50/50 confirmado: diseño caso-control. "
                 "Las probabilidades requerirán corrección de prevalencia "
                 f"(config.REAL_PREVALENCE = {C.REAL_PREVALENCE}).")

    report["filas_finales"] = int(len(df))
    out = C.CLEAN_DIR / "cleaned_data.csv"
    df.to_csv(out, index=False)
    U.save_json(report, C.CLEAN_DIR / "cleaning_log.json")
    log.banner("SALIDA")
    log.info(f"[ok] {out}  ({df.shape})")
    log.close()
    U.hard_exit()


if __name__ == "__main__":
    main()
