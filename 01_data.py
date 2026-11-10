# 01_data.py — clean OCELOT chromophore v1 and assign every molecule to
# train/val/test twice: random 80/10/10 and scaffold 80/10/10.
# Targets: ie_level = -VIE and ea_level = VEA (frontier energy levels from
# total-energy differences) plus HOMO and LUMO (orbital energies) for comparison.
# Output: data/ocelot_clean.parquet
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
SEED = 42
TARGETS = ["ie_level", "ea_level", "homo", "lumo"]

# ---- 1. load ----
df = pd.read_csv(ROOT / "ocelot.csv",
                 usecols=["identifier", "smiles", "homo", "lumo", "vie", "vea"])
df = df.rename(columns={"identifier": "mol_id"})
print(f"loaded:                    {len(df)}")

# ---- 2. energy-level targets ----
# -VIE: ionization energy as an energy level (counterpart of HOMO)
# VEA as stored: already on the same scale as LUMO (counterpart of LUMO)
df["ie_level"] = -df["vie"]
df["ea_level"] = df["vea"]
df = df.drop(columns=["vie", "vea"])

# ---- 3. canonical SMILES ----
df["smiles"] = df["smiles"].apply(Chem.CanonSmiles)

# ---- 4. drop every duplicated structure (labels disagree, so don't average) ----
df = df[~df["smiles"].duplicated(keep=False)].reset_index(drop=True)
print(f"after dropping duplicates: {len(df)}")

# ---- 5. random split 80 / 10 / 10 ----
train, rest = train_test_split(df.index, test_size=0.2, random_state=SEED)
val, test = train_test_split(rest, test_size=0.5, random_state=SEED)
df.loc[train, "split_random"] = "train"
df.loc[val, "split_random"] = "val"
df.loc[test, "split_random"] = "test"

# ---- 6. scaffold split 80 / 10 / 10 ----
df["scaffold"] = df["smiles"].apply(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s))

# one group of row indices per scaffold, largest group first
groups = sorted(df.groupby("scaffold").groups.values(), key=len, reverse=True)

n = len(df)
filled = 0
for idx in groups:
    if filled + len(idx) <= 0.8 * n:
        df.loc[idx, "split_scaffold"] = "train"
    elif filled + len(idx) <= 0.9 * n:
        df.loc[idx, "split_scaffold"] = "val"
    else:
        df.loc[idx, "split_scaffold"] = "test"
    filled += len(idx)

# ---- 7. report and save ----
for col in ["split_random", "split_scaffold"]:
    print(f"\n{col}:")
    print(df.groupby(col)[TARGETS].agg(["mean", "std"]).round(3).T)

df.to_parquet(DATA / "ocelot_clean.parquet", index=False)
print(f"\nsaved {len(df)} molecules -> {DATA / 'ocelot_clean.parquet'}")