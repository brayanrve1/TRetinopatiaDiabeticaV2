# -*- coding: utf-8 -*-
"""
PASO 06 — TRANSPORTEABILIDAD Y DESCOMPOSICIÓN DEL GAP (artefacto final)
========================================================================

  1. MODELO PRINCIPAL      : panel completo (paso 03), con su AUROC de CV y
                             de test hold-out + IC95 bootstrap.
  2. MODELO TRANSPORTABLE  : reentrenado aquí sobre el train usando SOLO las
                             variables con dato real en NHANES; se evalúa en
                             CV y test hold-out, se compara con DeLong contra
                             el principal y se guarda como .joblib final.

Salidas:
    outputs/models/<ganador>_transportable.joblib (+ _meta.json)
    outputs/external/gap_decomposition.json
    outputs/external/final_summary.json
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config as C
import utils as U


def main():
    U.ensure_dirs()
    log = U.StepLogger("06_transportability_gap")

    # ---- Datos y split idénticos a los pasos 03/04 ---------------------------
    df = U.load_cleaned(log)
    y = df[C.TARGET].astype(int)
    split = U.load_json(C.MODEL_DIR / "split.json")
    feats = split["features"]
    tr_ix = np.asarray(split["train_idx"])
    te_ix = np.asarray(split["test_idx"])
    Xtr, ytr = df.iloc[tr_ix][feats].astype(float), y.iloc[tr_ix]
    Xte, yte = df.iloc[te_ix][feats].astype(float), y.iloc[te_ix]
    log.info(f"[i] Train {len(Xtr)} | Test {len(Xte)} | panel: {len(feats)} features")

    # ---- Ganador del benchmark -------------------------------------------------
    cv_tbl = pd.read_csv(C.MODEL_DIR / "cv_metrics.csv")
    cv_tbl = cv_tbl[cv_tbl["estado"] == "ok"].sort_values("auc_cv", ascending=False)
    best = str(cv_tbl.iloc[0]["modelo"])
    auc_cv_20 = float(cv_tbl.iloc[0]["auc_cv"])
    meta = U.load_json(C.MODEL_DIR / f"{best}_meta.json")
    bp = {k.split("__")[-1]: v for k, v in meta["mejores_params"].items()}
    log.info(f"[i] Ganador: {best} | mejores parámetros: {bp}")

    # ---- Resultados externos del paso 05 -----------------------------------------
    ext_p = C.EXT_DIR / "external_summary.json"
    if not ext_p.exists():
        raise SystemExit("[!] Falta external_summary.json: ejecuta primero "
                         "05_external_validation.py")
    ext = U.load_json(ext_p)
    ext_metricas = ext["metricas"]
    ext20 = next((m["auroc"] for m in ext_metricas if "principal" in m["modelo"]),
                 ext_metricas[0]["auroc"])
    ext17 = next((m["auroc"] for m in ext_metricas
                  if "transportable" in m["modelo"]), ext_metricas[-1]["auroc"])
    etiqueta = ext.get("etiqueta_cohorte", "NHANES")

    # ---- Panel transportable (nativas según imputation_table.csv) -----------------
    imp = pd.read_csv(C.EXT_DIR / "imputation_table.csv")
    set_nat = set(imp.loc[imp["pct_faltante_nhanes"] < 99, "feature"])
    nativas = [f for f in feats if f in set_nat]
    excluidas = [f for f in feats if f not in set_nat]
    log.info(f"[i] Panel transportable ({len(nativas)}): {nativas}")
    log.info(f"[i] Excluidas por no existir en NHANES: {excluidas}")

    # ---- 1. Modelo principal (panel completo, interno) -------------------------------
    log.banner(f"1. MODELO PRINCIPAL — {best} con panel completo "
               f"({len(feats)} variables)")
    pipe20 = joblib.load(C.MODEL_DIR / f"{best}.joblib")
    p20 = pipe20.predict_proba(Xte)[:, 1]
    auc20 = float(roc_auc_score(yte, p20))
    lo20, hi20 = U.bootstrap_auc_ci(yte, p20)
    log.info(f"   AUC CV-interna = {auc_cv_20:.4f}")
    log.info(f"   AUC test = {auc20:.4f}  IC95[{lo20:.4f}, {hi20:.4f}]")

    # ---- 2. Modelo transportable (nativas, interno) -------------------------------------
    log.banner(f"2. MODELO TRANSPORTABLE — {best} con {len(nativas)} variables")
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    m17 = U.build_pipeline(best, U.make_model(best, gpu=False, **bp))
    aucs17 = cross_val_score(m17, Xtr[nativas], ytr, cv=cv,
                             scoring="roc_auc", n_jobs=1)
    auc_cv_17 = float(aucs17.mean())
    m17.fit(Xtr[nativas], ytr)
    p17 = m17.predict_proba(Xte[nativas])[:, 1]
    auc17 = float(roc_auc_score(yte, p17))
    lo17, hi17 = U.bootstrap_auc_ci(yte, p17)
    log.info(f"   AUC CV-interna = {auc_cv_17:.4f} ± {aucs17.std():.4f}")
    log.info(f"   AUC test = {auc17:.4f}  IC95[{lo17:.4f}, {hi17:.4f}]")

    joblib.dump(m17, C.MODEL_DIR / f"{best}_transportable.joblib")
    U.save_json({"modelo": f"{best}_transportable",
                 "mejores_params": meta["mejores_params"],
                 "auc_cv": auc_cv_17, "auc_test": auc17,
                 "ic95_test": [round(lo17, 4), round(hi17, 4)],
                 "variables": nativas,
                 "excluidas_por_indisponibles_externo": excluidas},
                C.MODEL_DIR / f"{best}_transportable_meta.json")
    log.info(f"   [ok] guardado: {best}_transportable.joblib (+ meta)")

    # ---- 3. DeLong interno -----------------------------------------------------------------
    log.banner("3. DELONG INTERNO (principal vs transportable, test hold-out)")
    zi, pi = U.delong_test(yte.values, p20, p17)
    sig = "SIGNIFICATIVA" if pi < 0.05 else "NO significativa"
    log.info(f"   z={zi:.2f} p={pi:.4f} -> diferencia {sig}")

    # ---- 4. Cascada y descomposición del gap ----------------------------------------------
    gap_panel = auc20 - auc17
    gap_dom = auc17 - float(ext17)
    gap_total = auc20 - float(ext20)
    log.banner("4. CASCADA DE RENDIMIENTO Y DESCOMPOSICIÓN DEL GAP")
    log.info(f"   AUC interno  panel {len(feats)} : {auc20:.4f} "
             f"IC[{lo20:.3f},{hi20:.3f}]")
    log.info(f"   AUC interno  panel {len(nativas)} : {auc17:.4f} "
             f"IC[{lo17:.3f},{hi17:.3f}]   (diferencia por panel   = {gap_panel:+.4f})")
    log.info(f"   AUC externo  panel {len(nativas)} : {ext17:.4f} en {etiqueta}   "
             f"(diferencia por dominio  = {gap_dom:+.4f})")
    log.info(f"   AUC externo  panel {len(feats)} : {ext20:.4f} (3 variables imputadas)")
    log.info("")
    log.info(f"   Degradación total interno->externo (panel completo): {gap_total:+.4f}")
    log.info(f"     -> atribuible a la reducción del panel: {gap_panel:+.4f}")
    log.info(f"     -> atribuible a dominio/desenlace:      {gap_dom:+.4f}")

    # ---- 5. Artefacto único para la tesis -----------------------------------------------------
    tri_ext = None
    tri_p = C.EXT_DIR / "triage_external.csv"
    if tri_p.exists():
        tri_ext = pd.read_csv(tri_p)
        tri_ext = tri_ext[tri_ext["sens_obj"] == C.TARGET_SENSITIVITY]
    triaje = {}
    if tri_ext is not None:
        for _, r in tri_ext.iterrows():
            clave = "principal" if "principal" in str(r["modelo"]) else "transportable"
            triaje[clave] = {
                "umbral": round(float(r["umbral"]), 4),
                "sensibilidad": float(r["sensibilidad"]),
                "especificidad": float(r["especificidad"]),
                "tasa_derivacion": float(r["tasa_derivacion"]),
                "PPV_prev_real": float(r["PPV_prev_real"]),
                "NPV_prev_real": float(r["NPV_prev_real"]),
            }

    U.save_json({
        "algoritmo": best,
        "modelo_principal": {
            "panel": len(feats), "auc_cv": auc_cv_20,
            "auc_test": round(auc20, 4),
            "ic95_test": [round(lo20, 4), round(hi20, 4)],
            "joblib": f"{best}.joblib",
        },
        "modelo_transportable": {
            "panel": len(nativas), "variables": nativas,
            "excluidas": excluidas,
            "auc_cv": auc_cv_17, "auc_test": round(auc17, 4),
            "ic95_test": [round(lo17, 4), round(hi17, 4)],
            "joblib": f"{best}_transportable.joblib",
        },
        "delong_interno": {"z": round(float(zi), 3), "p": round(float(pi), 4)},
        "validacion_externa": {
            "etiqueta": etiqueta, "n": ext.get("cohorte_n"),
            "prevalencia_rd": ext.get("prevalencia_rd"),
            "auroc_principal": ext20, "auroc_transportable": ext17,
            "delong_externo": ext.get("delong_principal_vs_transportable"),
            "triaje_sens_objetivo": triaje,
        },
        "descomposicion_gap": {
            "por_reduccion_panel": round(float(gap_panel), 4),
            "por_dominio_y_desenlace": round(float(gap_dom), 4),
            "total": round(float(gap_total), 4),
        },
    }, C.EXT_DIR / "final_summary.json")

    U.save_json({
        "cascada": [
            {"config": f"panel {len(feats)} interno", "auroc": round(auc20, 4)},
            {"config": f"panel {len(nativas)} interno", "auroc": round(auc17, 4)},
            {"config": f"panel {len(nativas)} externo ({etiqueta})",
             "auroc": round(float(ext17), 4)},
            {"config": f"panel {len(feats)} externo ({etiqueta})",
             "auroc": round(float(ext20), 4)},
        ],
        "degradacion_por_panel": round(float(gap_panel), 4),
        "degradacion_por_dominio": round(float(gap_dom), 4),
        "delong_interno": {"z": round(float(zi), 3), "p": round(float(pi), 4)},
    }, C.EXT_DIR / "gap_decomposition.json")

    log.banner("FIN")
    log.info(f"[ok] {best}_transportable.joblib en {C.MODEL_DIR}")
    log.info(f"[ok] gap_decomposition.json + final_summary.json en {C.EXT_DIR}")
    log.close()
    U.hard_exit()


if __name__ == "__main__":
    main()





