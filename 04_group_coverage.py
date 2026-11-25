# 04_group_coverage.py — sanity check of the group library before any SHAP.
# For each group: in how many molecules it occurs (all / random test set),
# what share of all heavy atoms it labels, and one example molecule.
from pathlib import Path

import pandas as pd
from rdkit import Chem

from lib import atom_groups

ROOT = Path(__file__).parent
DATA = ROOT / "data"

df = pd.read_parquet(DATA / "ocelot_clean.parquet")
is_test = (df["split_random"] == "test").to_numpy()

atom_counts = {}        # group -> number of atoms labelled with it
mol_all = {}            # group -> number of molecules containing it
mol_test = {}           # same, test set only
example = {}            # group -> first molecule containing it

for smiles, test in zip(df["smiles"], is_test):
    labels = atom_groups(Chem.MolFromSmiles(smiles))
    for g in labels:
        atom_counts[g] = atom_counts.get(g, 0) + 1
    for g in set(labels):
        mol_all[g] = mol_all.get(g, 0) + 1
        mol_test[g] = mol_test.get(g, 0) + test
        example.setdefault(g, smiles)

table = pd.DataFrame({
    "molecules": pd.Series(mol_all),
    "test molecules": pd.Series(mol_test),
    "atom share %": (pd.Series(atom_counts) / sum(atom_counts.values()) * 100).round(2),
    "example": pd.Series(example),
}).sort_values("molecules", ascending=False)

pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 200)
print(table.to_string())

print(df["smiles"].str.contains(r"\[N\+\]\(=O\)\[O-\]").sum(), df["smiles"].str.contains("S(=O)(=O)", regex=False).sum())