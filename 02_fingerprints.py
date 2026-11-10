# 02_fingerprints.py — collision-free Morgan fingerprints (CF-MF).
# Every distinct Morgan substructure identifier found in the TRAINING molecules
# gets its own column, so two different substructures never share a column.
# Built twice, because the random and scaffold splits have different training sets.
# Output: data/cfmf_{split}.npz               sparse matrix, rows in ocelot_clean.parquet order
#         data/cfmf_{split}_vectorizer.joblib  remembers which identifier is which column
from pathlib import Path

import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RADIUS = 2

df = pd.read_parquet(DATA / "ocelot_clean.parquet")
generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS)


def morgan_counts(smiles):
    """{identifier: count} for every circular substructure in one molecule (unfolded)."""
    mol = Chem.MolFromSmiles(smiles)
    fp = generator.GetSparseCountFingerprint(mol)
    return {str(k): v for k, v in fp.GetNonzeroElements().items()}


# ---- 1. fingerprint every molecule (same for both splits) ----
fps = df["smiles"].apply(morgan_counts)
print(f"distinct identifiers in whole dataset: {len(set().union(*fps))}")
print(f"substructures per molecule (median): {fps.apply(len).median():.0f}\n")

for split in ["split_random", "split_scaffold"]:
    name = split.replace("split_", "")

    # ---- 2. vocabulary from training molecules only ----
    is_train = (df[split] == "train").to_numpy()
    vectorizer = DictVectorizer()
    vectorizer.fit(fps[is_train])
    X = vectorizer.transform(fps)        # identifiers never seen in training are