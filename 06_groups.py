# 05_groups.py — how much does each functional group shift each target?
# Per-column SHAP values are computed for the val + test molecules (random split)
# and then summed per functional group (lib.group_totals).
# Done for Ridge (exact SHAP) and LightGBM (TreeSHAP).
# Output: data/group_shap_{model}_{target}.parquet   (rows = molecules, columns = groups)
#         results/groups_{model}_{target}.csv       (one row per group)
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from lib import group_totals

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
TARGETS = ["ie_level", "ea_level", "homo", "lumo"]
CHUNK = 200          # molecules processed at once (200 x 42k columns ~ 70 MB)
MIN_MOLECULES = 50   # report a group only if it occurs in at least this many molecules

df = pd.read_parquet(DATA / "ocelot_clean.parquet")
X = sparse.load_npz(DATA / "cfmf_random.npz").tocsr()
vocabulary = joblib.load(DATA / "cfmf_random_vectorizer.joblib").vocabulary_
train_rows = np.flatnonzero(df["split_random"] == "train")
eval_rows = np.flatnonzero(df["split_random"].isin(["val", "test"]))
x_mean = np.asarray(X[train_rows].mean(axis=0)).ravel()    # average count of each column


def ridge_shap(model, rows):
    """Exact SHAP for a linear model: weight * (count - average count)."""
    return (X[rows].toarray() - x_mean) * model.coef_


def lgbm_shap(model, rows):
    """TreeSHAP, computed by LightGBM itself. The last column is the base value."""
    dense = X[rows].toarray()                       # dense input: sparse input gave garbage
    contrib = model.predict(dense, pred_contrib=True)
    # check: SHAP values + base value must equal the prediction
    error = np.abs(contrib.sum(axis=1) - model.predict(dense)).max()
    assert error < 1e-6, f"TreeSHAP does not add up to the prediction (error {error:.1e})"
    return contrib[:, :-1]


explainers = {"lgbm": lgbm_shap}

for target in TARGETS:
    for name, shap_function in explainers.items():
        model = joblib.load(MODELS / f"{name}_random_{target}.joblib")
        print(f"\n===== {target} — {name} ({len(eval_rows)} molecules) =====", flush=True)

        # ---- per-molecule group totals, CHUNK molecules at a time ----
        records = []
        for start in range(0, len(eval_rows), CHUNK):
            chunk = eval_rows[start:start + CHUNK]
            phi = shap_function(model, chunk)
            for smiles, phi_row in zip(df["smiles"].iloc[chunk], phi):
                records.append(group_totals(smiles, phi_row, vocabulary))
            print(f"  {start + len(chunk)}/{len(eval_rows)} molecules", flush=True)
        totals = pd.DataFrame(records, index=df["mol_id"].iloc[eval_rows])
        totals.to_parquet(DATA / f"group_shap_{name}_{target}.parquet")

        # ---- sanity check (Ridge): group totals must add up to prediction - average prediction ----
        if name == "ridge":
            expected = model.predict(X[eval_rows]) - (model.intercept_ + x_mean @ model.coef_)
            error = np.abs(totals.sum(axis=1).to_numpy() - expected).max()
            print(f"check: largest difference between sum of groups and prediction = {error:.1e} eV")

        # ---- summary over the molecules that contain each group ----
        summary = totals.describe().T[["count", "mean", "25%", "50%", "75%"]]
        summary["frac_negative"] = (totals < 0).sum() / totals.notna().sum()
        summary = summary[summary["count"] >= MIN_MOLECULES]
        summary = summary.drop(index=["rest", "baseline"], errors="ignore").sort_values("50%")
        summary.index.name = "group"
        print(summary.round(3).to_string())
        summary.to_csv(RESULTS / f"groups_{name}_{target}.csv")