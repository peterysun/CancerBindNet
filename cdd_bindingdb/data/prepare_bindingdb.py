"""
Process raw BindingDB data into a clean training dataset.

Filters for cancer-relevant targets, parses messy IC50/Kd values,
validates SMILES with RDKit, and saves train-ready CSV files.

Accepts both .tsv and .zip input — zip files are read directly without
extraction.

Usage:
    python data/prepare_bindingdb.py --input data/raw/BindingDB_All_202605_tsv.zip
    python data/prepare_bindingdb.py --input data/raw/BindingDB_All.tsv --max-rows 500000
"""

import os
import sys
import json
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# Cancer targets we filter for. Keys are short labels we use internally;
# values are search keywords matched against BindingDB's "Target Name" column.
CANCER_TARGETS = {
    'EGFR':  ['egfr', 'epidermal growth factor receptor', 'erbb1', 'her1'],
    'VEGFR': ['vegfr', 'vascular endothelial growth factor receptor', 'kdr', 'flt1'],
    'IDH1':  ['idh1', 'isocitrate dehydrogenase 1'],
    'IDH2':  ['idh2', 'isocitrate dehydrogenase 2'],
    'BRAF':  ['braf', 'b-raf', 'serine/threonine-protein kinase b-raf'],
    'CDK2':  ['cdk2', 'cyclin-dependent kinase 2'],
    'ABL1':  ['abl1', 'abl', 'tyrosine-protein kinase abl'],
    'PARP1': ['parp1', 'parp-1', 'poly [adp-ribose] polymerase 1'],
    'ALK':   ['alk', 'anaplastic lymphoma kinase'],
    'MET':   [' met ', 'hepatocyte growth factor receptor', 'hgfr'],
}

ACTIVE_THRESHOLD_NM = 1000.0   # IC50 < 1 uM = active (industry standard)
MIN_EXAMPLES_PER_TARGET = 50


def open_bindingdb(input_path: Path):
    """Open BindingDB file. Auto-detects whether it's a .zip or .tsv."""
    if str(input_path).endswith('.zip'):
        print(f"  Reading TSV directly from zip...")
        zf = zipfile.ZipFile(input_path, 'r')
        tsv_files = [f for f in zf.namelist() if f.endswith('.tsv')]
        if not tsv_files:
            raise ValueError(f"No .tsv inside {input_path}. Contents: {zf.namelist()}")
        print(f"  Found inside zip: {tsv_files[0]}")
        return zf.open(tsv_files[0])
    else:
        print(f"  Opening TSV...")
        return open(input_path, 'rb')


def parse_ic50(value_str) -> float:
    """
    Parse messy IC50 strings into floats.
    BindingDB stores values like '5.2', '>10000', '<0.1', '1.2e3'.
    """
    if pd.isna(value_str) or str(value_str).strip() == '':
        return np.nan
    s = str(value_str).strip()
    for ch in '><~=':
        s = s.replace(ch, '')
    try:
        val = float(s.strip())
        if val < 0.001 or val > 1e8:
            return np.nan
        return val
    except ValueError:
        return np.nan


def assign_target(target_name: str) -> str:
    """Match a BindingDB target name to one of our cancer target labels."""
    if pd.isna(target_name):
        return None
    name_lower = str(target_name).lower()
    for label, keywords in CANCER_TARGETS.items():
        for kw in keywords:
            if kw in name_lower:
                return label
    return None


def validate_smiles(smiles_list: list) -> list:
    """Use RDKit to filter out invalid SMILES strings."""
    try:
        from rdkit import Chem
        results = []
        for smi in smiles_list:
            if pd.isna(smi) or str(smi).strip() == '':
                results.append(False)
            else:
                results.append(Chem.MolFromSmiles(str(smi)) is not None)
        return results
    except ImportError:
        return [True] * len(smiles_list)


def prepare_dataset(input_path: Path, output_dir: Path, max_rows: int = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nStep 1: Reading {input_path.name}")
    needed_cols = ['Ligand SMILES', 'Target Name', 'IC50 (nM)', 'Kd (nM)', 'Ki (nM)']
    file_obj = open_bindingdb(input_path)

    try:
        df = pd.read_csv(
            file_obj, sep='\t', encoding='utf-8',
            usecols=lambda c: any(nc in c for nc in needed_cols),
            low_memory=False, nrows=max_rows,
        )
    except Exception:
        # Some BindingDB exports have non-utf8 chars in metadata fields
        file_obj = open_bindingdb(input_path)
        df = pd.read_csv(
            file_obj, sep='\t', encoding='latin1',
            usecols=lambda c: any(nc in c for nc in needed_cols),
            low_memory=False, nrows=max_rows,
        )

    df.columns = [c.strip() for c in df.columns]
    print(f"  Loaded {len(df):,} rows")

    # Locate the columns we need (names may have extra text)
    smiles_col = next((c for c in df.columns if 'SMILES' in c), None)
    target_col = next((c for c in df.columns if 'Target Name' in c), None)
    ic50_col   = next((c for c in df.columns if 'IC50' in c), None)
    kd_col     = next((c for c in df.columns if 'Kd' in c and 'nM' in c), None)
    ki_col     = next((c for c in df.columns if 'Ki' in c and 'nM' in c), None)

    if not smiles_col or not target_col:
        print(f"ERROR: Missing required columns. Available: {list(df.columns)}")
        sys.exit(1)

    # Filter to cancer targets
    print(f"\nStep 2: Filtering for cancer targets...")
    tqdm.pandas(desc="  Matching")
    df['target_label'] = df[target_col].progress_apply(assign_target)
    df = df[df['target_label'].notna()].copy()
    print(f"  Kept {len(df):,} rows")
    print(f"\n  Target distribution:")
    print(df['target_label'].value_counts().to_string())

    if len(df) == 0:
        print("ERROR: No rows matched cancer targets.")
        sys.exit(1)

    # Parse binding affinity values
    print(f"\nStep 3: Parsing affinity values...")
    if ic50_col: df['ic50_nM'] = df[ic50_col].apply(parse_ic50)
    if kd_col:   df['kd_nM']  = df[kd_col].apply(parse_ic50)
    if ki_col:   df['ki_nM']  = df[ki_col].apply(parse_ic50)

    # Prefer IC50, fall back to Kd, then Ki
    df['affinity_nM'] = np.nan
    if ic50_col: df['affinity_nM'] = df.get('ic50_nM', np.nan)
    if kd_col:   df.loc[df['affinity_nM'].isna(), 'affinity_nM'] = df.get('kd_nM', np.nan)
    if ki_col:   df.loc[df['affinity_nM'].isna(), 'affinity_nM'] = df.get('ki_nM', np.nan)

    df = df[df['affinity_nM'].notna()].copy()

    # Convert to log scale for training stability
    df['pIC50'] = (-np.log10(df['affinity_nM'] / 1e9)).clip(2, 12)
    df['active'] = (df['affinity_nM'] < ACTIVE_THRESHOLD_NM).astype(int)
    print(f"  Valid: {len(df):,} | Active: {df['active'].sum():,} ({df['active'].mean()*100:.1f}%)")

    # Validate SMILES with RDKit
    print(f"\nStep 4: Validating SMILES...")
    df['smiles'] = df[smiles_col].astype(str).str.strip()
    valid = validate_smiles(df['smiles'].tolist())
    df = df[valid].copy()
    print(f"  Valid SMILES: {len(df):,}")

    # If a molecule was tested multiple times, keep the lowest IC50
    df = df.sort_values('affinity_nM')
    df = df.drop_duplicates(subset=['smiles', 'target_label'], keep='first')
    print(f"  After dedup: {len(df):,}")

    # Drop targets with too few examples (won't train well)
    counts = df['target_label'].value_counts()
    valid_targets = counts[counts >= MIN_EXAMPLES_PER_TARGET].index
    df = df[df['target_label'].isin(valid_targets)].copy()

    # Save
    print(f"\nStep 5: Saving...")
    final = df[['smiles', 'target_label', 'affinity_nM', 'pIC50', 'active']].reset_index(drop=True)
    final.to_csv(output_dir / 'bindingdb_cancer.csv', index=False)
    print(f"  Combined dataset: {len(final):,} rows")

    for target in final['target_label'].unique():
        t_df = final[final['target_label'] == target]
        t_df.to_csv(output_dir / f'{target.lower()}_data.csv', index=False)
        print(f"  {target}: {len(t_df):,} rows ({t_df['active'].mean()*100:.1f}% active)")

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump({
            'total': len(final),
            'targets': final['target_label'].value_counts().to_dict(),
            'active_rate': float(final['active'].mean()),
        }, f, indent=2)

    print(f"\nDone. Next: python scripts/train.py --target EGFR")
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path,
                        default=Path(__file__).parent / 'raw' / 'BindingDB_All_202605_tsv.zip')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent / 'processed')
    parser.add_argument('--max-rows', type=int, default=None)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    prepare_dataset(args.input, args.output, args.max_rows)
