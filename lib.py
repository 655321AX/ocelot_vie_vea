# lib.py — functional group definitions and per-molecule SHAP-to-group assignment.
# Used by 04_group_coverage.py and 05_groups.py.
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

RADIUS = 2   # must be the same as in 02_fingerprints.py

# Functional groups as SMARTS, most specific first.
# Atoms marked ":1" are the group's own atoms; unmarked atoms are only context
# (e.g. the carbon an OH sits on) and do NOT belong to the group.
GROUPS = [
    # ---- strong acceptor units (must come before nitrile / alkene / carbonyl) ----
    ("dicyanovinyl",       "[#6:1]=[#6:1]([CX2:1]#[NX1:1])[CX2:1]#[NX1:1]"),
    ("imide",              "[#6X3:1](=[OX1:1])[#7X3:1][#6X3:1]=[OX1:1]"),
    ("nitro",              "[#7:1](~[OX1:1])~[OX1:1]"),
    ("sulfone",            "[#16X4:1](=[OX1:1])=[OX1:1]"),
    ("sulfoxide",          "[#16X3:1]=[OX1:1]"),
    # ---- carbonyl family ----
    ("amide",              "[#6X3:1](=[OX1:1])[#7X3:1]"),
    ("carboxylic acid",    "[CX3:1](=[OX1:1])[OX2H1:1]"),
    ("ester",              "[#6X3:1](=[OX1:1])[OX2:1][#6]"),
    ("aldehyde",           "[CX3H1:1]=[OX1:1]"),
    ("ketone",             "[#6][CX3:1](=[OX1:1])[#6]"),
    ("other C=O",          "[#6:1]=[OX1:1]"),
    ("thioamide",          "[#6X3:1](=[SX1:1])[#7X3:1]"),
    ("thiocarbonyl",       "[#6:1]=[SX1:1]"),
    # ---- other multiple-bond groups ----
    ("nitrile",            "[CX2:1]#[NX1:1]"),
    ("azo",                "[NX2:1]=[NX2:1]"),
    ("imine",              "[CX3:1]=[NX2:1]"),
    # ---- aromatic heteroatoms ----
    ("thiophene-type S",   "[s:1]"),
    ("furan-type O",       "[o:1]"),
    ("pyrrole-type N",     "[nX3:1]"),
    ("pyridine-type N",    "[nX2:1]"),
    # ---- O, N, S substituents ----
    ("phenol OH",          "[OX2H1:1]c"),
    ("alcohol OH",         "[OX2H1:1][#6]"),
    ("aryl alkyl ether",   "[OX2:1](c)[CX4]"),
    ("ether O",            "[OX2:1]([#6])[#6]"),
    ("primary amine",      "[NX3H2:1][#6]"),
    ("secondary amine",    "[NX3H1:1]([#6])[#6]"),
    ("tertiary amine",     "[NX3H0:1]([#6])([#6])[#6]"),
    ("thioether S",        "[SX2:1]([#6])[#6]"),
    # ---- halogens ----
    ("fluorine",           "[F:1]"),
    ("chlorine",           "[Cl:1]"),
    ("bromine",            "[Br:1]"),
    ("iodine",             "[I:1]"),
    # ---- rare elements (any bonding) ----
    ("selenium",           "[#34:1]"),
    ("tellurium",          "[#52:1]"),
    ("silicon",            "[#14:1]"),
    ("boron",              "[#5:1]"),
    ("phosphorus",         "[#15:1]"),
    ("arsenic",            "[#33:1]"),
    # ---- carbon framework (catches what no group above took) ----
    ("alkyne",             "[CX2:1]#[CX2:1]"),
    ("alkene",             "[CX3:1]=[CX3:1]"),
    ("aromatic C",         "[c:1]"),
    ("other sp2 C",        "[#6X3:1]"),
    ("sp3 C",              "[CX4:1]"),
    ("other N",            "[#7:1]"),
    ("other O",            "[#8:1]"),
    ("other S",            "[#16:1]"),
]

# Pre-compile each pattern once and note which pattern positions are group atoms.
PATTERNS = []
for name, smarts in GROUPS:
    pattern = Chem.MolFromSmarts(smarts)
    assert pattern is not None, f"invalid SMARTS for {name}: {smarts}"
    core = [atom.GetIdx() for atom in pattern.GetAtoms() if atom.GetAtomMapNum() == 1]
    PATTERNS.append((name, pattern, core))

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS)


def atom_groups(mol):
    """Group name for every atom; 'rest' if the atom belongs to no group.
    If an atom fits several groups, the one listed first in GROUPS wins."""
    labels = ["rest"] * mol.GetNumAtoms()
    for name, pattern, core in PATTERNS:
        for match in mol.GetSubstructMatches(pattern):
            for position in core:
                atom = match[position]
                if labels[atom] == "rest":
                    labels[atom] = name
    return labels


def occurrences(mol, vocabulary):
    """Every fingerprint occurrence in one molecule as (column, centre atom, count of that column).
    Substructures never seen in training have no column and are skipped."""
    info = rdFingerprintGenerator.AdditionalOutput()
    info.AllocateBitInfoMap()
    GENERATOR.GetSparseCountFingerprint(mol, additionalOutput=info)

    result = []
    for identifier, places in info.GetBitInfoMap().items():
        col = vocabulary.get(str(identifier))
        if col is None:
            continue
        for centre_atom, radius in places:
            result.append((col, centre_atom, len(places)))
    return result


def group_totals(smiles, phi, vocabulary):
    """Split one molecule's per-column SHAP values (phi) over its functional groups.

    Every fingerprint occurrence belongs to the group of its centre atom IN THIS
    molecule. A column that occurs twice gives half of its SHAP value to each.
    SHAP of columns the molecule does not contain goes to 'baseline'.
    Returns {group: total SHAP in eV}; groups absent from the molecule are left out.
    """
    mol = Chem.MolFromSmiles(smiles)
    labels = atom_groups(mol)

    totals = defaultdict(float)
    used_columns = set()
    for col, centre_atom, count in occurrences(mol, vocabulary):
        totals[labels[centre_atom]] += phi[col] / count
        used_columns.add(col)

    totals["baseline"] = phi.sum() - phi[list(used_columns)].sum()
    return dict(totals)


def group_columns(smiles, vocabulary):
    """{group: {column: number of that column's occurrences centred on the group's atoms}}.
    Used by the removal test: subtracting these counts removes exactly the
    occurrences that group_totals gives to the group."""
    mol = Chem.MolFromSmiles(smiles)
    labels = atom_groups(mol)

    result = defaultdict(lambda: defaultdict(int))
    for col, centre_atom, count in occurrences(mol, vocabulary):
        result[labels[centre_atom]][col] += 1
    return result