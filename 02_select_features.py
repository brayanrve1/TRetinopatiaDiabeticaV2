# -*- coding: utf-8 -*-
"""
PASO 02 — SELECCIÓN DE FEATURES (embudo multi-criterio, solo con datos de TRAIN)

Salidas: outputs/features/selected_features.json + feature_funnel.csv + log
"""
import numpy as np
import pandas as pd

import config as C
import utils as U


def stage_flags(df):
    """Flags binarios candidatos según prevalencia."""
    lo, hi = C.SEL_FLAG_PREVALENCE
    out = []
    for c in df.columns:
        if c in (C.TARGET, C.ID_COL) or c in C.NUMERIC_CANDIDATES:
            continue
        s = df[c]
        if s.isna().mean() > 0.05 or s.nunique(dropna=True) != 2:
            continue
        prev = s.mean()
        if lo <= prev <= hi:
            out.append(c)
    return out


def main():
    U.ensure_dirs()
    log = U.StepLogger("02_select_features")

    df = U.load_cleaned(log)
    y = df[C.TARGET].astype(int)
    tr_ix, te_ix = U.get_split_indices(df, y)
    dft, yt = df.iloc[tr_ix], y.iloc[tr_ix]
    log.info(f"[i] Selección calculada SOLO en train: n={len(dft)} "
             f"(test n={len(te_ix)} permanece intocado)")

    look_ahead = [c for c in C.LOOKAHEAD_EXCLUDE if c in dft.columns]
    candidates = ([c for c in C.NUMERIC_CANDIDATES if c in dft.columns]
                  + stage_flags(dft))
    candidates = [c for c in candidates
                  if c not in look_ahead and c not in C.DROP_ALWAYS]
    log.info(f"[i] Candidatas iniciales: {len(candidates)} "
             f"(excluidas por look-ahead: {look_ahead})")

    funnel = {c: dict(etapa_final="S0", motivo="candidata") for c in candidates}
    alive = list(candidates)

    # --- S1 disponibilidad -------------------------------------------------
    miss = dft[alive].isna().mean()
    drop = [c for c in alive if miss[c] > C.SEL_MAX_MISSING]
    for c in drop:
        funnel[c] = dict(etapa_final="S1_disponibilidad",
                         motivo=f"faltantes {miss[c]:.1%} > {C.SEL_MAX_MISSING:.0%}")
    alive = [c for c in alive if c not in drop]
    log.banner("S1 DISPONIBILIDAD")
    log.info(f"   eliminadas: {drop if drop else 'ninguna'}  -> quedan {len(alive)}")

    # --- S2 varianza ---------------------------------------------------------
    drop = []
    for c in alive:
        nu = dft[c].nunique(dropna=True)
        top = dft[c].value_counts(normalize=True).max() if nu else 1
        if nu <= 1 or (nu == 2 and top > 0.995):
            drop.append(c)
    for c in drop:
        funnel[c] = dict(etapa_final="S2_varianza", motivo="constante/casi-constante")
    alive = [c for c in alive if c not in drop]
    log.banner("S2 VARIANZA")
    log.info(f"   eliminadas: {drop if drop else 'ninguna'}  -> quedan {len(alive)}")

    # --- S3 redundancia -------------------------------------------------------
    log.banner("S3 REDUNDANCIA (|rho| > umbral)")
    num_alive = [c for c in alive if dft[c].nunique() > 2]
    rho = dft[num_alive].corr(method="spearman")
    cluster_drop = set()
    for i in range(len(num_alive)):
        for j in range(i + 1, len(num_alive)):
            a, b = num_alive[i], num_alive[j]
            if a in cluster_drop or b in cluster_drop:
                continue
            v = rho.loc[a, b]
            if pd.notna(v) and abs(v) > C.SEL_CORR_THRESHOLD:
                # se queda la de mayor asociación univariada con la clase
                ra = abs(U.rank_biserial(dft.loc[yt == 0, a].dropna(),
                                         dft.loc[yt == 1, a].dropna()))
                rb = abs(U.rank_biserial(dft.loc[yt == 0, b].dropna(),
                                         dft.loc[yt == 1, b].dropna()))
                loser = b if (ra if not np.isnan(ra) else 0) >= (rb if not np.isnan(rb) else 0) else a
                cluster_drop.add(loser)
                log.info(f"   {a} <-> {b}: rho={v:.3f} -> se elimina '{loser}'")
    for c in cluster_drop:
        funnel[c] = dict(etapa_final="S3_redundancia",
                         motivo=f"|rho|>{C.SEL_CORR_THRESHOLD} con otra feature")
    alive = [c for c in alive if c not in cluster_drop]
    log.info(f"   -> quedan {len(alive)}")

    # --- S4 efecto univariado ---------------------------------------------------
    log.banner("S4 EFECTO UNIVARIADO")
    effects = {}
    drop = []
    core = set(C.SEL_CORE_KEEP)
    for c in alive:
        if dft[c].nunique() <= 2:
            e = U.cramers_v(dft[c].dropna(), yt[dft[c].notna()])
        else:
            e = U.rank_biserial(dft.loc[yt == 0, c].dropna(),
                                dft.loc[yt == 1, c].dropna())
        e = abs(e) if pd.notna(e) else 0.0
        effects[c] = round(e, 4)
        if e < C.SEL_MIN_EFFECT and c not in core:
            drop.append(c)
    for c in drop:
        funnel[c] = dict(etapa_final="S4_efecto",
                         motivo=f"efecto {effects[c]} < {C.SEL_MIN_EFFECT}")
    alive = [c for c in alive if c not in drop]
    top_eff = sorted(effects.items(), key=lambda kv: -kv[1])[:15]
    log.info(f"   eliminadas por efecto bajo: {drop if drop else 'ninguna'}")
    log.info(f"   top 15 efectos: {top_eff}")
    log.info(f"   -> quedan {len(alive)}")

    # --- S5 importancia de modelo (permutación, CV) --------------------------------
    log.banner("S5 IMPORTANCIA POR PERMUTACIÓN (XGBoost + CV)")
    mean_imp = {}
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import roc_auc_score

        Xa = dft[alive].astype(float)
        skf = StratifiedKFold(3, shuffle=True, random_state=C.SEED)
        acc = {c: [] for c in alive}
        for tr, va in skf.split(Xa, yt):
            m = U.make_model("xgboost", gpu=C.USE_GPU,
                             max_depth=4, learning_rate=0.08, n_estimators=250)
            try:
                m.fit(Xa.iloc[tr], yt.iloc[tr])
            except Exception:
                m = U.make_model("xgboost", gpu=False,
                                 max_depth=4, learning_rate=0.08, n_estimators=250)
                m.fit(Xa.iloc[tr], yt.iloc[tr])
            pi = permutation_importance(m, Xa.iloc[va], yt.iloc[va],
                                        n_repeats=5, random_state=C.SEED,
                                        scoring="roc_auc", n_jobs=C.N_JOBS)
            for k, c in enumerate(alive):
                acc[c].append(pi.importances_mean[k])
        mean_imp = {c: float(np.mean(v)) for c, v in acc.items()}
        drop = [c for c in alive
                if mean_imp[c] <= 0 and c not in core and effects.get(c, 0) < 0.10]
        for c in drop:
            funnel[c] = dict(etapa_final="S5_importancia",
                             motivo=f"importancia media {mean_imp[c]:.4f} <= 0")
        alive = [c for c in alive if c not in drop]
        log.info(f"   eliminadas por importancia nula: {drop if drop else 'ninguna'}")
        imp_tbl = pd.DataFrame({"feature": mean_imp.keys(),
                                "importancia_perm": mean_imp.values(),
                                "efecto_univariado": [effects.get(c, np.nan)
                                                      for c in mean_imp]})
        imp_tbl = imp_tbl.sort_values("importancia_perm", ascending=False)
        imp_tbl.to_csv(C.FEAT_DIR / "importance_selection.csv", index=False)
        log.info("\n" + imp_tbl.head(20).to_string(index=False))
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Etapa S5 omitida ({type(e).__name__}: {e}); "
                 f"la selección queda en S4.")

    # --- S6 tope de tamaño del panel ---------------------------------------------
    if C.SEL_MAX_FEATURES is not None and len(alive) > C.SEL_MAX_FEATURES:
        log.banner(f"S6 TOPE DE PANEL (máx {C.SEL_MAX_FEATURES} features)")
        # ranking combinado: 60% importancia de modelo + 40% efecto univariado
        imp = pd.Series({c: mean_imp.get(c, float("nan")) for c in alive})
        eff = pd.Series({c: effects.get(c, 0.0) for c in alive})
        rank_imp = imp.rank(ascending=False, na_option="bottom")
        rank_eff = eff.rank(ascending=False)
        score = (0.6 * rank_imp + 0.4 * rank_eff).sort_values()
        keep = list(score.index[:C.SEL_MAX_FEATURES])
        # garantiza el núcleo clínico aunque quede fuera del top
        for c in C.SEL_CORE_KEEP:
            if c in alive and c not in keep:
                for i in range(len(keep) - 1, -1, -1):
                    if keep[i] not in C.SEL_CORE_KEEP:
                        log.info(f"   [core] entra '{c}', sale '{keep[i]}'")
                        keep[i] = c
                        break
        dropped = [c for c in alive if c not in keep]
        for c in dropped:
            funnel[c] = dict(etapa_final="S6_tope",
                             motivo=f"fuera del top-{C.SEL_MAX_FEATURES} combinado")
        alive = keep
        log.info(f"   recortadas {len(dropped)}: {dropped}")
        log.info(f"   -> quedan {len(alive)}")

    # --- resultado -------------------------------------------------------------
    log.banner("RESULTADO")
    for c in alive:
        funnel[c] = dict(etapa_final="SELECCIONADA",
                         motivo=f"efecto={effects.get(c, np.nan)}, supera embudo")
    log.info(f"Features finales: {len(alive)}")
    log.info(str(alive))
    log.info(f"Guardadas para el modelo (excluidas del modelo final por diseño): "
             f"look-ahead={look_ahead}, Tier-2={C.TIER2_DIAG}")

    U.save_json({"features": alive, "lookahead_excluidas": look_ahead,
                 "tier2_solo_diagnostico": C.TIER2_DIAG,
                 "core_keep": C.SEL_CORE_KEEP,
                 "n_train": int(len(dft)), "n_test": int(len(te_ix))},
                C.FEAT_DIR / "selected_features.json")
    pd.DataFrame([dict(feature=k, **v) for k, v in funnel.items()]).to_csv(
        C.FEAT_DIR / "feature_funnel.csv", index=False)
    log.info(f"[ok] selected_features.json + feature_funnel.csv en {C.FEAT_DIR}")
    log.close()
    U.hard_exit()


if __name__ == "__main__":
    main()
