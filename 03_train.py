# 03_train.py — mean baseline, Ridge and LightGBM on CF-MF, both splits, all targets.
# Ridge alpha and the number of LightGBM trees are chosen on the validation set;
# the test set is used once at the end.
# Output: models/{model}_{split}_{target}.joblib, data/results_test.csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
SEED = 42
TARGETS = ["ie_level", "ea_level", "homo", "lumo"]

RIDGE_ALPHAS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]

# max_depth=12: without a depth limit, trees on fingerprints grow 200+ levels deep
# and TreeSHAP becomes numerically unstable (found on QM9).
LGBM_PARAMS = dict(n_estimators=15000, learning_rate=0.05, max_depth=12, num_leaves=63,
                   min_child_samples=10, subsample=0.85, subsample_freq=1,
                   colsample_bytree=0.7, reg_lambda=1, random_state=SEED, verbose=-1)

df = pd.read_parquet(DATA / "ocelot_clean.parquet")
results = []

for split in ["random", "scaffold"]:
    X = sparse.load_npz(DATA / f"cfmf_{split}.npz").tocsr()
    tr = (df[f"split_{split}"] == "train").to_numpy()
    va = (df[f"split_{split}"] == "val").to_numpy()
    te = (df[f"split_{split}"] == "test").to_numpy()

    for target in TARGETS:
        y = df[target].to_numpy()
        print(f"\n===== {split} / {target} =====")

        # ---- Ridge: try each alpha, keep the one with the lowest validation MAE ----
        best_mae = np.inf
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha).fit(X[tr], y[tr])
            mae = mean_absolute_error(y[va], model.predict(X[va]))
            if mae < best_mae:
                best_mae, ridge = mae, model
        print(f"Ridge:    alpha = {ridge.alpha}, val MAE = {best_mae:.4f} eV")
        if ridge.alpha in (min(RIDGE_ALPHAS), max(RIDGE_ALPHAS)):
            print("          WARNING: best alpha is at the edge of the grid")

        # ---- LightGBM: add trees until the validation MAE stops improving ----
        lgbm = LGBMRegressor(**LGBM_PARAMS)
        lgbm.fit(X[tr], y[tr], eval_X=X[va], eval_y=y[va], eval_metric="l1",
                 callbacks=[early_stopping(200, verbose=False), log_evaluation(1000)])
        print(f"LightGBM: {lgbm.best_iteration_} trees, "
              f"val MAE = {lgbm.best_score_['valid_0']['l1']:.4f} eV")

        # ---- save models ----
        joblib.dump(ridge, MODELS / f"ridge_{split}_{target}.joblib")
        joblib.dump(lgbm, MODELS / f"lgbm_{split}_{target}.joblib")

        # ---- evaluate once on the test set ----
        candidates = {
            "mean": np.full(te.sum(), y[tr].mean()),   # always predict the training mean
            "ridge": ridge.predict(X[te]),
            "lgbm": lgbm.predict(X[te]),
        }
        for name, pred in candidates.items():
            results.append({
                "split": split, "target": target, "model": name,
                "MAE": mean_absolute_error(y[te], pred),
                "RMSE": np.sqrt(mean_squared_error(y[te], pred)),
                "R2": r2_score(y[te], pred),
            })

# ---- report and save ----
results = pd.DataFrame(results)
print("\n========== TEST SET: MAE (eV) ==========")
print(results.pivot_table(index=["split", "model"], columns="target", values="MAE")
      [TARGETS].round(3))
print("\n========== TEST SET: R2 ==========")
print(results.pivot_table(index=["split", "model"], columns="target", values="R2")
      [TARGETS].round(2))
results.to_csv(DATA / "results_test.csv", index=False)
print(f"\nsaved models -> {MODELS}, results -> {DATA / 'results_test.csv'}")