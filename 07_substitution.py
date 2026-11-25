# 06_substitution.py — what does a halogen atom do to the IE and EA levels?
# For every halogen atom in the val + test molecules: replace it by H, predict again,
# effect = prediction(with halogen) - prediction(with H).
# If the H-analogue exists in the dataset, the real DFT difference is recorded too.
# Output: results/substitution_{target}.csv   (one row per substitution)
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
ELEMENTS = ["F", "Cl", "Br", "I"]

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


# ---- 1. one variant per halogen atom: that atom replaced by H ----
rows, elements, variants, pair_rows = [], [], [], []
for row in eval_rows:
    mol = Chem.MolFromSmiles(df["smiles"].iloc[row])
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ELEMENTS:
            continue
        new = Chem.RWMol(mol)
        new.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(1)   # halogen -> H
        new = Chem.RemoveHs(new)                              # H becomes an ordinary implicit H
        rows.append(row)
        elements.append(atom.GetSymbol())
        variants.append(counts(new))
        pair_rows.append(smiles_to_row.get(Chem.MolToSmiles(new), -1))   # -1 = not in dataset

rows, elements, pair_rows = np.array(rows), np.array(elements), np.array(pair_rows)
has_pair = pair_rows >= 0
X_sub = vectorizer.transform(variants)
print(f"{len(rows)} substitutions, {has_pair.sum()} with the H-analogue in the dataset\n")

# ---- 2. effect per substitution, both models, plus DFT where available ----
for target in TARGETS:
    y = df[target].to_numpy()
    table = pd.DataFrame({"element": elements})
    for name in ["ridge", "lgbm"]:
        model = joblib.load(MODELS / f"{name}_random_{target}.joblib")
        table[name] = model.predict(X[rows]) - model.predict(X_sub)
    table["dft"] = np.where(has_pair, y[rows] - y[np.where(has_pair, pair_rows, 0)], np.nan)
    table.to_csv(RESULTS / f"substitution_{target}.csv", index=False)

    # ---- 3. summary per element ----
    g = table.groupby("element")
    summary = pd.DataFrame({
        "n": g.size(),
        "ridge": g["ridge"].median(),
        "lgbm": g["lgbm"].median(),
        "lgbm_frac_neg": g["lgbm"].apply(lambda v: (v < 0).mean()),
        "n_pairs": g["dft"].count(),
        "dft_pairs": g["dft"].median(),
        "lgbm_pairs": table[has_pair].groupby("element")["lgbm"].median(),
    }).loc[ELEMENTS]
    print(f"===== {target}: effect of halogen vs H (eV, median) =====")
    print(summary.round(3).to_string(), "\n")