"""Download all IC50 measurements for human EGFR (CHEMBL203) from ChEMBL.

Saves one row per measurement to data/egfr_raw.csv with columns:
    molecule_chembl_id, smiles, standard_relation, ic50_nM

standard_relation says whether the value is exact ("=") or a limit
(">", "<", ">=", ...), e.g. ">10000" means "no effect up to 10 µM".

Run from the project folder with:
    uv run python scripts/download_egfr_ic50.py
"""

from pathlib import Path

import pandas as pd
from chembl_webresource_client.new_client import new_client

TARGET_ID = "CHEMBL203"  # Epidermal growth factor receptor, Homo sapiens
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "egfr_raw.csv"


def fetch_activities() -> pd.DataFrame:
    """Ask ChEMBL for every IC50 record on EGFR, already converted to nM."""
    query = new_client.activity.filter(
        target_chembl_id=TARGET_ID,
        standard_type="IC50",
        standard_units="nM",
    ).only(
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_relation",
        "standard_value",
    )
    print(f"ChEMBL reports {len(query)} matching IC50 records; downloading...")
    return pd.DataFrame.from_records(list(query))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, make IC50 numeric, and drop rows missing key fields."""
    df = df.rename(
        columns={"canonical_smiles": "smiles", "standard_value": "ic50_nM"}
    )[["molecule_chembl_id", "smiles", "standard_relation", "ic50_nM"]]
    df["ic50_nM"] = pd.to_numeric(df["ic50_nM"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["molecule_chembl_id", "smiles", "ic50_nM"])
    print(f"Dropped {before - len(df)} rows missing a SMILES or IC50 value.")
    return df.reset_index(drop=True)


def main() -> None:
    df = clean(fetch_activities())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(
        f"Saved {len(df)} rows "
        f"({df['molecule_chembl_id'].nunique()} unique molecules) to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
