"""Turn raw ChEMBL EGFR IC50 records into one labelled row per molecule.

Reads  data/egfr_raw.csv      (from scripts/download_egfr_ic50.py)
Writes data/egfr_labelled.csv with columns:
    molecule_chembl_id, smiles, ic50_nM (median), n_measurements,
    value_type ("exact" or "lower_limit"), label

Run from the project folder with:
    uv run python scripts/clean_and_label.py
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "egfr_raw.csv"
OUTPUT = DATA / "egfr_labelled.csv"

ACTIVE_CUTOFF_NM = 1000  # IC50 below 1 µM counts as active
MAX_PLAUSIBLE_LIMIT_NM = 1_000_000  # '>' values above 1 mM are likely entry errors


def keep_trustworthy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep exact values, plus '>' limits that prove inactivity and are plausible."""
    exact = df["standard_relation"] == "="
    greater = df["standard_relation"] == ">"
    implausible = greater & (df["ic50_nM"] > MAX_PLAUSIBLE_LIMIT_NM)
    clearly_inactive = (
        greater & (df["ic50_nM"] >= ACTIVE_CUTOFF_NM) & ~implausible
    )
    kept = df[exact | clearly_inactive]
    print(
        f"Step 1: kept {len(kept)} of {len(df)} measurements "
        f"({exact.sum()} exact, {clearly_inactive.sum()} '>' inactive limits; "
        f"dropped {implausible.sum()} '>' values above 1 mM)"
    )
    return kept


def prefer_exact(df: pd.DataFrame) -> pd.DataFrame:
    """Per molecule: use only '=' rows if it has any, else its '>' rows."""
    is_exact = df["standard_relation"] == "="
    has_exact = is_exact.groupby(df["molecule_chembl_id"]).transform("any")
    kept = df[is_exact | ~has_exact]
    print(
        f"Step 2: ignored {len(df) - len(kept)} '>' values from molecules "
        f"that also have exact measurements"
    )
    return kept


def merge_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Combine repeated measurements of a molecule using the median IC50."""
    merged = (
        df.groupby("molecule_chembl_id")
        .agg(
            smiles=("smiles", "first"),
            ic50_nM=("ic50_nM", "median"),
            n_measurements=("ic50_nM", "size"),
            value_type=("standard_relation", "first"),
        )
        .reset_index()
    )
    # After prefer_exact a molecule's rows are all '=' or all '>'.
    merged["value_type"] = merged["value_type"].map(
        {"=": "exact", ">": "lower_limit"}
    )
    print(f"Step 3: {len(df)} measurements -> {len(merged)} unique molecules")
    return merged


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label each molecule active (IC50 < 1000 nM) or inactive."""
    df["label"] = (df["ic50_nM"] < ACTIVE_CUTOFF_NM).map(
        {True: "active", False: "inactive"}
    )
    return df


def main() -> None:
    raw = pd.read_csv(RAW)
    labelled = add_labels(
        merge_duplicates(prefer_exact(keep_trustworthy_rows(raw)))
    )
    labelled.to_csv(OUTPUT, index=False)

    counts = labelled["label"].value_counts()
    total = len(labelled)
    print(f"Step 4: saved {total} molecules to {OUTPUT}")
    for name in ("active", "inactive"):
        n = counts.get(name, 0)
        print(f"  {name:8} {n:6}  ({n / total:.1%})")


if __name__ == "__main__":
    main()
