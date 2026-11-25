# 07_sulfur_swap.py — what does aromatic S do compared with O and NH?
# For every aromatic S atom in the val + test molecules: swap it for O (thiophene -> furan)
# and, separately, for NH (thiophene -> pyrrole), then predict again.
# effect = prediction(with S) - prediction(with O or NH).
# If the swapped molecule exists in the dataset, the real DFT difference is recorded too.
# Output: results/sulfur_swap_{target}.csv   (one row per swap)
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
TARGETS = ["ie_level", "ea_level"]
SWAPS = {"S->O": 8, "S->NH": 7}      # label: new atomic number

df = pd.read_parquet(DATA / "ocelot_clean.parquet")
X = sparse.load_npz(DATA / "cfmf_random.npz").tocsr()
vectorizer = joblib.load(DATA / "cfmf_random_vectorizer.joblib")
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)   # same as 02_fingerprints.py
eval_rows = np.flatnonzero(df["split_random"].isin(["val", "test"]))
smiles_to_row = {s: i for i, s in enumerate(df["smiles"])}


def counts(mol):
    """{identifier: count} for one molecule, exactly as in 02_fingerprints.py."""
    fp = generator.GetSparseCountFingerprint(mol)
    return {str(k): v for k, v in fp.GetNonzeroElements().items()}


# ---- 1. one variant per aromatic S atom and per swap ----
rows, swaps, variants, pair_rows = [], [], [], []
failed = 0
for row in eval_rows:
    mol = Chem.MolFromSmiles(df["smiles"].iloc[row])
    for atom in mol.GetAtoms():
        if not (atom.GetSymbol() == "S" and atom.GetIsAromatic()):
            continue
        for label, atomic_number in SWAPS.items():
            new = Chem.RWMol(mol)
            swapped = new.GetAtomWithIdx(atom.GetIdx())
            swapped.SetAtomicNum(atomic_number)
            if label == "S->NH":
                swapped.SetNumExplicitHs(1)       # pyrrole-type N carries an H
            try:
                Chem.SanitizeMol(new)             # rejects swaps that give an impossible structure
            except Exception:
                failed += 1
                continue
            rows.append(row)
            swaps.append(label)
            variants.append(counts(new))
            pair_rows.append(smiles_to_row.get(Chem.MolToSmiles(new), -1))   # -1 = not in dataset

rows, swaps, pair_rows = np.array(rows), np.array(swaps), np.array(pair_rows)
has_pair = pair_rows >= 0
X_sub = vectorizer.transform(variants)
print(f"{len(rows)} swaps ({failed} rejected as invalid), "
      f"{has_pair.sum()} with the swapped molecule in the dataset\n")

# ---- 2. effect per swap, both models, plus DFT where available ----
for target in TARGETS:
    y = df[target].to_numpy()
    table = pd.DataFrame({"swap": swaps})
    for name in ["ridge", "lgbm"]:
        model = joblib.load(MODELS / f"{name}_random_{target}.joblib")
        table[name] = model.predict(X[rows]) - model.predict(X_sub)
    table["dft"] = np.where(has_pair, y[rows] - y[np.where(has_pair, pair_rows, 0)], np.nan)
    table.to_csv(RESULTS / f"sulfur_swap_{target}.csv", index=False)

    # ---- 3. summary per swap ----
    g = table.groupby("swap")
    summary = pd.DataFrame({
        "n": g.size(),
        "ridge": g["ridge"].median(),
        "lgbm": g["lgbm"].median(),
        "lgbm_frac_neg": g["lgbm"].apply(lambda v: (v < 0).mean()),
        "n_pairs": g["dft"].count(),
        "dft_pairs": g["dft"].median(),
        "lgbm_pairs": table[has_pair].groupby("swap")["lgbm"].median(),
    })
    print(f"===== {target}: effect of S vs swapped atom (eV, median) =====")
    print(summary.round(3).to_string(), "\n")