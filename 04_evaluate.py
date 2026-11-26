# -*- coding: utf-8 -*-
"""
PASO 04 — EVALUACIÓN COMPLETA DE TRIAJE (archivo único)
Incluye, en orden:
  1. Métricas de todos los modelos en test hold-out (AUROC + IC bootstrap,
     AUPRC, Brier, ECE). El ganador se elige por AUC de CV (el test NO
     participa en la elección).
  2. Test de DeLong pareado vs el ganador.
  3. ABLACIÓN DE FEATURES: curva AUC vs nº de features con ranking derivado
     solo de train (permutación out-of-fold) y paneles anidados top-K;
     calcula el punto de saturación (K para >=99% y >=99.5% del máximo,
     y K dentro del IC95 del máximo).
  4. Tabla de triaje (sensibilidad fija) + cohorte de despliegue simulada
     a prevalencia real.
  5. Calibración, ROC y decision-curve de todos los modelos.
  6. Subgrupos de equidad (ganador).
  7. SHAP del ganador (si está disponible).
Salidas: outputs/eval/* (tablas, figuras, log).
"""
import glob
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

import config as C
import utils as U

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def load_models():
    models = {}
    for p in glob.glob(str(C.MODEL_DIR / "*.joblib")):
        name = os.path.basename(p).replace(".joblib", "")
        meta_p = C.MODEL_DIR / f"{name}_meta.json"
        if not meta_p.exists():
            continue
        models[name] = dict(pipe=joblib.load(p), meta=U.load_json(meta_p))
    return models


def choose_best(models, metrics, log):
    """Ganador = mayor AUC de CV (el test no se usa para elegir)."""
    cv_tbl = pd.read_csv(C.MODEL_DIR / "cv_metrics.csv")
    cv_tbl = cv_tbl[cv_tbl["estado"] == "ok"][["modelo", "auc_cv"]]
    joined = cv_tbl.merge(metrics[["modelo", "auroc"]], on="modelo")
    joined = joined.sort_values(["auc_cv", "auroc"], ascending=False)
    best = joined.iloc[0]["modelo"]
    log.info("Regla de selección: mayor AUC de CV (el test no participa en la elección)")
    log.info(joined.to_string(index=False))
    log.info(f"\n   >> Ganador: {best}")
    return best


def ablation_section(df, y, tr_ix, te_ix, best, log):
    """Curva AUC vs número de features + punto de saturación."""
    log.banner("ABLACIÓN DE FEATURES (¿cuántas variables son suficientes?)")
    f = pd.read_csv(C.FEAT_DIR / "feature_funnel.csv")
    pool = f[f["etapa_final"].isin(["SELECCIONADA", "S6_tope"])]["feature"].tolist()
    pool = [c for c in pool if c in df.columns]
    log.info(f"[i] Pool de supervivientes del embudo (sin tope): {len(pool)}")

    dft, yt = df.iloc[tr_ix], y.iloc[tr_ix]
    Xte, yte = df.iloc[te_ix], y.iloc[te_ix]

    # ranking SOLO con train (permutación out-of-fold, XGBoost)
    log.info("[i] Ranking derivado solo de train (permutación out-of-fold)...")
    X = dft[pool].astype(float)
    gpu = U.probe_gpu("xgboost", log) if C.USE_GPU else False
    skf = StratifiedKFold(3, shuffle=True, random_state=C.SEED)
    acc = {c: [] for c in pool}
    for tr, va in skf.split(X, yt):
        m = U.make_model("xgboost", gpu=gpu, max_depth=4,
                         learning_rate=0.08, n_estimators=250)
        try:
            m.fit(X.iloc[tr], yt.iloc[tr])
        except Exception:
            m = U.make_model("xgboost", gpu=False, max_depth=4,
                             learning_rate=0.08, n_estimators=250)
            m.fit(X.iloc[tr], yt.iloc[tr])
        pi = permutation_importance(m, X.iloc[va], yt.iloc[va], n_repeats=5,
                                    random_state=C.SEED, scoring="roc_auc",
                                    n_jobs=C.N_JOBS)
        for k, c in enumerate(pool):
            acc[c].append(pi.importances_mean[k])
    mean_imp = {c: float(np.mean(v)) for c, v in acc.items()}
    ranking = sorted(pool, key=lambda c: -mean_imp[c])
    log.info("Top-20 del ranking: " + str(ranking[:20]))
    pd.DataFrame({"feature": ranking,
                  "importancia_oof": [mean_imp[c] for c in ranking]}).to_csv(
        C.EVAL_DIR / "ablation_ranking.csv", index=False)

    meta = U.load_json(C.MODEL_DIR / f"{best}_meta.json")
    bp = {k.split("__")[-1]: v for k, v in meta["mejores_params"].items()}
    gpu_b = (U.probe_gpu(best, log) if C.USE_GPU and best in
             ("xgboost", "lightgbm", "catboost") else False)
    log.info(f"[i] Algoritmo de la ablación: {best} (ganador del benchmark)")

    ks = sorted({int(k) for k in C.ABLATION_KS if int(k) <= len(ranking)})
    rows = []
    for k in ks:
        feats_k = ranking[:k]
        m = U.build_pipeline(best, U.make_model(best, gpu=gpu_b, **bp))
        m.fit(dft[feats_k].astype(float), yt)
        p = m.predict_proba(Xte[feats_k].astype(float))[:, 1]
        auc = roc_auc_score(yte, p)
        lo, hi = U.bootstrap_auc_ci(yte, p)
        rows.append(dict(K=k, auroc=round(auc, 4), ic_lo=round(lo, 4),
                         ic_hi=round(hi, 4)))
        note = (f"   (K=1 = solo '{ranking[0]}')"
                if k == 1 and ranking[0] == C.SINGLE_FEATURE_BASELINE else "")
        log.info(f"   K={k:>3}  AUROC={auc:.4f}  IC95[{lo:.4f},{hi:.4f}]{note}")
    curve = pd.DataFrame(rows)
    curve.to_csv(C.EVAL_DIR / "ablation_curve.csv", index=False)

    imax = curve["auroc"].idxmax()
    max_auc = float(curve.loc[imax, "auroc"])
    k_at_max = int(curve.loc[imax, "K"])
    log.info(f"\n   AUC máximo: {max_auc:.4f} con K={k_at_max}")
    summary = {"auc_maximo": max_auc, "K_del_maximo": k_at_max}
    for target in C.ABLATION_TARGETS:
        hit = curve[curve["auroc"] >= target * max_auc]
        k_star = int(hit["K"].min()) if len(hit) else None
        summary[f"K_para_{target}"] = k_star
        if k_star is not None:
            auc_k = float(curve[curve["K"] == k_star]["auroc"].iloc[0])
            log.info(f"   K mínimo para >= {target:.1%} del máximo: {k_star} "
                     f"(AUC {auc_k:.4f}, dif {auc_k - max_auc:+.4f})")
        else:
            log.info(f"   K mínimo para >= {target:.1%} del máximo: NO ALCANZADO")
    ci_lo_max = float(curve.loc[imax, "ic_lo"])
    within = curve[curve["auroc"] >= ci_lo_max]
    if len(within):
        k_ci = int(within["K"].min())
        summary["K_dentro_IC95_del_maximo"] = k_ci
        log.info(f"   K mínimo dentro del IC95 del máximo: {k_ci} "
                 f"(panel bajo costo estadísticamente equivalente)")

    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.errorbar(curve["K"], curve["auroc"],
                    yerr=[curve["auroc"] - curve["ic_lo"],
                          curve["ic_hi"] - curve["auroc"]],
                    fmt="o-", color="#2c6e49", capsize=3, label="AUC test")
        for target, ls in zip(C.ABLATION_TARGETS, ("--", ":")):
            ax.axhline(target * max_auc, color="gray", ls=ls, lw=1)
            ax.text(curve["K"].max() * 1.01, target * max_auc,
                    f"{target:.1%} del máx", va="center", fontsize=8)
            k_s = summary.get(f"K_para_{target}")
            if k_s:
                ax.axvline(k_s, color="#b5651d", ls=":", lw=1)
        ax.set_xlabel("Número de features (top-K del ranking)")
        ax.set_ylabel("AUROC en test hold-out")
        ax.set_title(f"Ablación de features — {best} "
                     f"(máx {max_auc:.3f} en K={k_at_max})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(C.FIG_DIR / "feature_ablation.png", dpi=130)
        plt.close(fig)
        log.info("[i] Figura: figs/feature_ablation.png")

    U.save_json({**summary, "ranking_top20": ranking[:20], "modelo": best},
                C.EVAL_DIR / "ablation_summary.json")
    return curve, summary


def main():
    U.ensure_dirs()
    log = U.StepLogger("04_evaluate")

    df = U.load_cleaned(log)
    split = U.load_json(C.MODEL_DIR / "split.json")
    feats = split["features"]
    y = df[C.TARGET].astype(int)
    tr_ix, te_ix = np.asarray(split["train_idx"]), np.asarray(split["test_idx"])
    Xte, yte = df.iloc[te_ix][feats].astype(float), y.iloc[te_ix]

    models = load_models()
    if not models:
        raise RuntimeError("No hay modelos entrenados; ejecuta 03_train_models.py")
    log.info(f"[i] Modelos evaluados: {list(models)}")
    preds = {n: m["pipe"].predict_proba(Xte)[:, 1] for n, m in models.items()}

    # ---------- 1. métricas ---------------------------------------------------
    log.banner("1. MÉTRICAS EN TEST HOLD-OUT")
    rows = []
    for name, p in preds.items():
        auc = roc_auc_score(yte, p)
        lo, hi = U.bootstrap_auc_ci(yte, p)
        rows.append(dict(modelo=name, auroc=round(auc, 4),
                         ic95=f"[{lo:.4f}, {hi:.4f}]",
                         auprc=round(average_precision_score(yte, p), 4),
                         brier=round(brier_score_loss(yte, p), 4),
                         ece=round(U.expected_calibration_error(yte, p), 4)))
    metrics = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    metrics.to_csv(C.EVAL_DIR / "model_metrics.csv", index=False)
    log.info(metrics.to_string(index=False))

    best = choose_best(models, metrics, log)

    # ---------- 2. DeLong ------------------------------------------------------
    log.banner(f"2. TEST DE DELONG (vs {best})")
    dl_rows = []
    for name, p in preds.items():
        if name == best:
            continue
        z, pv = U.delong_test(yte.values, preds[best], p)
        dl_rows.append(dict(modelo=name, z=round(z, 3), p_valor=round(pv, 4),
                            significativo="sí" if pv < 0.05 else "no"))
        log.info(f"   {best} vs {name:<14} z={z:.2f} p={pv:.4f}")
    pd.DataFrame(dl_rows).to_csv(C.EVAL_DIR / "delong_tests.csv", index=False)

    # ---------- 3. ablación de features -------------------------------------------
    curve, ab_summary = ablation_section(df, y, tr_ix, te_ix, best, log)

    # ---------- 4. tabla de triaje ---------------------------------------------------
    log.banner(f"4. TABLA DE TRIAJE (sensibilidad fija; despliegue a prevalencia "
               f"{C.REAL_PREVALENCE:.0%})")
    tri_all = []
    for name, p in preds.items():
        t = U.triage_metrics(yte, p)
        t.insert(0, "modelo", name)
        tri_all.append(t)
        log.info(f"\n[{name}]\n{t.to_string(index=False)}")
    tri = pd.concat(tri_all)
    tri.to_csv(C.EVAL_DIR / "triage_thresholds.csv", index=False)

    row90 = tri[(tri["modelo"] == best) & (tri["sens_obj"] == C.TARGET_SENSITIVITY)]
    thr90 = float(row90.iloc[0]["umbral"]) if len(row90) else None
    if thr90 is not None:
        dep = U.deployment_cohort_metrics(yte, preds[best], thr90)
        U.save_json(dep, C.EVAL_DIR / "deployment_cohort.json")
        log.info("\nCOHORTE DE DESPLIEGUE SIMULADA (ganador, umbral sens>=90%)")
        log.info(pd.Series(dep).to_string())

    # ---------- 5. figuras --------------------------------------------------------------
    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        for name, p in preds.items():
            pt, pp = calibration_curve(yte, p, n_bins=8, strategy="quantile")
            ax.plot(pp, pt, "o-", label=name, ms=4)
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("Probabilidad predicha (espacio 50/50)")
        ax.set_ylabel("Frecuencia observada")
        ax.set_title("Calibración — todos los modelos (test)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(C.FIG_DIR / "calibration_all.png", dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        for name, p in preds.items():
            fpr, tpr, _ = roc_curve(yte, p)
            ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(yte, p):.3f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("1 - Especificidad"); ax.set_ylabel("Sensibilidad")
        ax.set_title("ROC — todos los modelos (test)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(C.FIG_DIR / "roc_all.png", dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        tt = np.linspace(0.01, 0.99, 199)
        for name, p in preds.items():
            _, nb, _ = U.net_benefit(yte, p, tt)
            ax.plot(tt, nb, label=name)
        _, _, nb_all = U.net_benefit(yte, preds[best], tt)
        ax.plot(tt, nb_all, "k--", label="derivar a todos")
        ax.axhline(0, color="gray", ls=":")
        ax.set_xlabel("Umbral"); ax.set_ylabel("Beneficio neto")
        ax.set_title("Decision Curve Analysis (test, prevalencia de muestra)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(C.FIG_DIR / "decision_curve_all.png", dpi=130)
        plt.close(fig)
        log.info("\n[i] Figuras: roc_all.png, calibration_all.png, decision_curve_all.png")

    # ---------- 6. subgrupos ----------------------------------------------------------------
    log.banner("6. SUBGRUPOS DE EQUIDAD (ganador, en test)")
    p_best = preds[best]
    sg_rows = []
    for label, (col, op, val) in C.SUBGROUPS.items():
        if col not in df.columns:
            continue
        s = df.iloc[te_ix][col]
        mask = {"lt": s < val, "ge": s >= val, "eq": s == val}[op].values
        if mask.sum() < C.SUBGROUP_MIN_N or len(np.unique(yte.values[mask])) < 2:
            continue
        yy = yte.values[mask]
        auc = roc_auc_score(yy, p_best[mask])
        if thr90 is not None:
            pred = (p_best[mask] >= thr90).astype(int)
            sens = float(((pred == 1) & (yy == 1)).sum() / max((yy == 1).sum(), 1))
            spec = float(((pred == 0) & (yy == 0)).sum() / max((yy == 0).sum(), 1))
        else:
            sens = spec = np.nan
        sg_rows.append(dict(subgrupo=label, n=int(mask.sum()), auroc=round(auc, 4),
                            sensibilidad_en_umbral=round(sens, 3),
                            especificidad_en_umbral=round(spec, 3)))
        log.info(f"   {label:<16} n={int(mask.sum()):>4}  AUROC={auc:.4f}  "
                 f"sens={sens:.3f} spec={spec:.3f}")
    pd.DataFrame(sg_rows).to_csv(C.EVAL_DIR / "subgroup_analysis.csv", index=False)

    # ---------- 7. SHAP ------------------------------------------------------------------------
    log.banner("7. SHAP (ganador)")
    try:
        import shap
        sub = Xte.sample(n=min(C.SHAP_MAX_SAMPLES, len(Xte)), random_state=C.SEED)
        model_raw = models[best]["pipe"].named_steps["clf"]
        explainer = shap.TreeExplainer(model_raw)
        sv = explainer.shap_values(sub)
        if isinstance(sv, list):
            sv = sv[1]
        sh = pd.Series(np.abs(sv).mean(axis=0), index=sub.columns)
        sh.sort_values(ascending=False).to_csv(C.EVAL_DIR / "shap_importance.csv")
        log.info("Top-15 |SHAP| medio:\n" + sh.sort_values(ascending=False)
                 .head(15).round(4).to_string())
        if HAS_PLOT:
            shap.summary_plot(sv, sub, show=False, max_display=15)
            fig = plt.gcf()
            fig.set_size_inches(9, 7)
            fig.tight_layout()
            fig.savefig(C.FIG_DIR / "shap_summary.png", dpi=140, bbox_inches="tight")
            plt.close("all")
            log.info("[i] Figura: shap_summary.png")
    except Exception as e:  # noqa: BLE001
        log.info(f"[i] SHAP no disponible o no aplica ({type(e).__name__}: {e}). "
                 f"Instala con: pip install shap")

    log.banner("FIN")
    log.info(f"Ganador: {best} | selección por CV. Resultados en {C.EVAL_DIR}")
    log.close()
    U.hard_exit()  # evita el crash de cierre de DLLs nativas en Windows


if __name__ == "__main__":
    main()
