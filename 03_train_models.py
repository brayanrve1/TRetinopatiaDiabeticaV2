# -*- coding: utf-8 -*-
"""
PASO 03 — ENTRENAMIENTO DEL BENCHMARK (6 modelos, protocolo idéntico)
- Mismo split estratificado para todos (seed fijada en config)
- Tuning por GridSearchCV con CV sobre el train; test INTACTO
- GPU (RTX) donde el backend lo permita, con fallback automático a CPU
Salidas: outputs/models/<nombre>.joblib + metadata + cv_metrics.csv + log
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold

import config as C
import utils as U


def main():
    U.ensure_dirs()
    log = U.StepLogger("03_train_models")

    df = U.load_cleaned(log)
    sel = U.load_json(C.FEAT_DIR / "selected_features.json")
    feats = [f for f in sel["features"] if f in df.columns]
    if not feats:
        raise RuntimeError("No hay features seleccionadas; ejecuta 02_select_features.py")
    log.info(f"[i] Features usadas: {len(feats)}")

    y = df[C.TARGET].astype(int)
    tr_ix, te_ix = U.get_split_indices(df, y)
    Xtr, ytr = df.iloc[tr_ix][feats].astype(float), y.iloc[tr_ix]
    Xte, yte = df.iloc[te_ix][feats].astype(float), y.iloc[te_ix]
    U.save_json({"train_idx": tr_ix.tolist(), "test_idx": te_ix.tolist(),
                 "features": feats}, C.MODEL_DIR / "split.json")
    log.info(f"[i] Train {len(Xtr)} ({ytr.mean():.1%} RD) | Test {len(Xte)} "
             f"({yte.mean():.1%} RD) — split guardado en models/split.json")

    backends = U.available_backends()
    log.info(f"[i] Backends: {backends} | USE_GPU={C.USE_GPU}")

    rows = []
    for name in C.MODELS_ENABLED:
        log.banner(f"MODELO: {name}")
        if name in ("xgboost", "lightgbm", "catboost") and backends.get(name) is None:
            log.info(f"[!] librería '{name}' no instalada -> se omite "
                     f"(pip install {name})")
            rows.append(dict(modelo=name, estado="omitido (no instalado)",
                             auc_cv=np.nan, auc_cv_std=np.nan,
                             mejores_params="", device=""))
            continue
        try:
            gs, device = U.fit_gs_or_fallback(name, Xtr, ytr, log)
        except Exception as e:  # noqa: BLE001
            log.info(f"[!] {name} falló: {e}")
            rows.append(dict(modelo=name, estado="error", auc_cv=np.nan,
                             auc_cv_std=np.nan, mejores_params="", device=""))
            continue

        # estimación honesta: CV con los mejores parámetros
        best_pipe = gs.best_estimator_
        cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
        aucs = cross_val_score(best_pipe, Xtr, ytr, cv=cv, scoring="roc_auc",
                               n_jobs=1 if name == "catboost" else C.N_JOBS)
        log.info(f"[i] {name}: AUC CV-{C.CV_FOLDS} = {aucs.mean():.4f} ± {aucs.std():.4f}")

        # re-fit final sobre todo el train con los mejores parámetros
        final = U.build_pipeline(name, U.make_model(name, gpu=(device == "GPU")))
        final.set_params(**{k.replace("clf__", "clf__"): v
                            for k, v in gs.best_params_.items()})
        final.fit(Xtr, ytr)

        joblib.dump(final, C.MODEL_DIR / f"{name}.joblib")
        U.save_json({"modelo": name, "device": device,
                     "mejores_params": gs.best_params_,
                     "auc_cv_interna": float(gs.best_score_),
                     "auc_cv": float(aucs.mean()), "auc_cv_std": float(aucs.std()),
                     "features": feats},
                    C.MODEL_DIR / f"{name}_meta.json")
        rows.append(dict(modelo=name, estado="ok", auc_cv=round(aucs.mean(), 4),
                         auc_cv_std=round(aucs.std(), 4),
                         mejores_params=str(gs.best_params_), device=device))

    tbl = pd.DataFrame(rows)
    tbl.to_csv(C.MODEL_DIR / "cv_metrics.csv", index=False)
    log.banner("RESUMEN CV")
    log.info(tbl.to_string(index=False))
    log.close()
    U.hard_exit()


if __name__ == "__main__":
    main()
