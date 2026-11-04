"""Turn each labelled EGFR molecule into a Morgan fingerprint, and draw examples.

Reads   data/egfr_labelled.csv          (from scripts/clean_and_label.py)
Writes  data/egfr_fingerprints.npz      X  : (n_molecules, 2048) array of 0/1 bits
                                        y  : 1 = active, 0 = inactive
                                        molecule_chembl_id, smiles : row order
        figures/example_molecules.png   very active, inactive, erlotinib

Run from the project folder with:
    uv run python scripts/make_fingerprints.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

ROOT = Path(__file__).resolve().parent.parent
LABELLED = ROOT / "data" / "egfr_labelled.csv"
FP_OUTPUT = ROOT / "data" / "egfr_fingerprints.npz"
FIG_OUTPUT = ROOT / "figures" / "example_molecules.png"

RADIUS = 2      # look at each atom's neighbourhood up to 2 bonds away
N_BITS = 2048   # length of the fingerprint
ERLOTINIB_ID = "CHEMBL553"

RDLogger.DisableLog("rdApp.*")  # we count unreadable SMILES ourselves


def parse_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """Turn SMILES text into RDKit molecules; drop the ones RDKit can't read."""
    df = df.copy()
    df["mol"] = df["smiles"].map(Chem.MolFromSmiles)
    bad = df["mol"].isna()
    if bad.any():
        print(f"Skipped {bad.sum()} SMILES RDKit could not read:")
        for _, row in df[bad].iterrows():
            print(f"  {row['molecule_chembl_id']}: {row['smiles']}")
    else:
        print("All SMILES were read successfully.")
    return df[~bad].reset_index(drop=True)


def make_fingerprints(mols: pd.Series) -> np.ndarray:
    """Morgan fingerprint for each molecule, as a row of 0s and 1s."""
    generator = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array(
        [generator.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8
    )


def pick_examples(df: pd.DataFrame) -> list[tuple[Chem.Mol, str]]:
    """Most potent molecule, a clear-cut inactive one, and erlotinib.

    Both picks need at least 2 measurements. The inactive pick must have a
    median of 10 µM-1 mM: a few ChEMBL '>' entries are ~1 M (impossible,
    likely unit errors), so the single highest IC50 is not a real inactive.
    Among those, take the most-measured molecule.
    """
    repeated = df[df["n_measurements"] >= 2]
    plausible_inactive = repeated[repeated["ic50_nM"].between(1e4, 1e6)]
    examples = [
        ("Very active", repeated.loc[repeated["ic50_nM"].idxmin()]),
        ("Inactive", plausible_inactive.loc[plausible_inactive["n_measurements"].idxmax()]),
    ]
    erlotinib = df[df["molecule_chembl_id"] == ERLOTINIB_ID]
    if erlotinib.empty:
        print(f"Erlotinib ({ERLOTINIB_ID}) is not in the data; drawing 2 molecules.")
    else:
        examples.append(("Erlotinib", erlotinib.iloc[0]))

    picked = []
    for title, row in examples:
        legend = (
            f"{title}: {row['molecule_chembl_id']}\n"
            f"IC50 = {row['ic50_nM']:g} nM ({row['label']}, n={row['n_measurements']})"
        )
        print(f"  {legend.replace(chr(10), '  ')}")
        picked.append((row["mol"], legend))
    return picked


def main() -> None:
    df = parse_molecules(pd.read_csv(LABELLED))

    X = make_fingerprints(df["mol"])
    y = (df["label"] == "active").astype(np.uint8).to_numpy()
    FP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FP_OUTPUT,
        X=X,
        y=y,
        molecule_chembl_id=df["molecule_chembl_id"].to_numpy(),
        smiles=df["smiles"].to_numpy(),
    )
    print(
        f"Saved fingerprints for {X.shape[0]} molecules "
        f"({X.shape[1]} bits each; {y.sum()} active, {len(y) - y.sum()} inactive) "
        f"to {FP_OUTPUT}"
    )
    print(f"On average {X.sum(axis=1).mean():.0f} of {N_BITS} bits are switched on.")

    print("Drawing examples:")
    mols, legends = zip(*pick_examples(df))
    image = Draw.MolsToGridImage(
        list(mols), molsPerRow=len(mols), subImgSize=(450, 400),
        legends=list(legends),
    )
    FIG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIG_OUTPUT)
    print(f"Saved image to {FIG_OUTPUT}")


if __name__ == "__main__":
    main()
