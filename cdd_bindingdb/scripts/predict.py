"""
Score new molecules using a trained model.

Usage:
    python scripts/predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python scripts/predict.py --smiles-file my_compounds.txt
    python scripts/predict.py --smiles "..." --model-dir runs/EGFR_20260510_044343
"""

import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.molecule_utils import smiles_to_features, passes_lipinski
from models.binding_model import BindingAffinityNet


def interpret_pic50(pic50: float) -> str:
    """Convert pIC50 to a readable summary."""
    ic50_nM = 10 ** (9 - pic50)
    if pic50 >= 8:
        verdict = "STRONG binder - excellent drug candidate"
    elif pic50 >= 7:
        verdict = "GOOD binder - drug-like potency"
    elif pic50 >= 6:
        verdict = "MODERATE binder - worth investigating"
    elif pic50 >= 5:
        verdict = "WEAK binder - needs optimization"
    else:
        verdict = "VERY WEAK - unlikely to be useful"
    return f"pIC50={pic50:.2f}  (IC50 ~ {ic50_nM:.1f} nM)  -> {verdict}"


def predict_smiles(smiles_list: list, model_dir: Path):
    """Run prediction on a list of SMILES strings."""
    nn_path = model_dir / 'nn_model.pt'
    rf_path = model_dir / 'rf_model.pkl'

    if not nn_path.exists() and not rf_path.exists():
        print(f"ERROR: No model in {model_dir}. Run train.py first.")
        sys.exit(1)

    print(f"\nScoring {len(smiles_list)} molecule(s)...")
    print("-" * 60)

    results = []
    for i, smiles in enumerate(smiles_list):
        smiles = smiles.strip()
        if not smiles:
            continue

        features = smiles_to_features(smiles)
        if features is None:
            print(f"[{i+1}] INVALID SMILES: {smiles}")
            continue

        lipinski = passes_lipinski(smiles)

        nn_pred = None
        if nn_path.exists():
            checkpoint = torch.load(nn_path, weights_only=False, map_location='cpu')
            model = BindingAffinityNet(input_dim=features.shape[0])
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            with torch.no_grad():
                x = torch.from_numpy(features).float().unsqueeze(0)
                nn_pred = model(x).item()

        rf_pred = None
        if rf_path.exists():
            with open(rf_path, 'rb') as f:
                rf = pickle.load(f)
            rf_pred = rf.predict(features.reshape(1, -1))[0]

        final = nn_pred if nn_pred is not None else rf_pred

        print(f"\n[{i+1}] {smiles[:60]}{'...' if len(smiles) > 60 else ''}")
        if nn_pred is not None:
            print(f"     NN:  {interpret_pic50(nn_pred)}")
        if rf_pred is not None:
            print(f"     RF:  {interpret_pic50(rf_pred)}")
        print(f"     Lipinski drug-like: {'yes' if lipinski else 'no'}")

        results.append({
            'smiles': smiles,
            'nn_pic50': nn_pred,
            'rf_pic50': rf_pred,
            'final_pic50': final,
            'lipinski': lipinski,
        })

    # Print ranking if more than one molecule
    if len(results) > 1:
        results.sort(key=lambda x: x['final_pic50'] or 0, reverse=True)
        print("\n--- Ranking (best to worst) ---")
        for rank, r in enumerate(results, 1):
            pic50 = r['final_pic50']
            ic50 = 10 ** (9 - pic50)
            print(f"  #{rank}: pIC50={pic50:.2f} (IC50~{ic50:.1f}nM)  {r['smiles'][:50]}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str, default=None)
    parser.add_argument('--smiles-file', type=Path, default=None)
    parser.add_argument('--model-dir', type=Path, default=None)
    args = parser.parse_args()

    # Auto-find most recent training run if not specified
    if args.model_dir is None:
        runs_dir = Path(__file__).parent.parent / 'runs'
        if runs_dir.exists():
            run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if run_dirs:
                args.model_dir = run_dirs[0]
                print(f"Using most recent model: {args.model_dir.name}")

    if args.model_dir is None or not args.model_dir.exists():
        print("ERROR: No trained model found. Run train.py first.")
        sys.exit(1)

    smiles_list = []
    if args.smiles:
        smiles_list.append(args.smiles)
    if args.smiles_file and args.smiles_file.exists():
        smiles_list.extend(args.smiles_file.read_text().strip().split('\n'))

    if not smiles_list:
        # Demo with known EGFR inhibitors
        print("No SMILES provided. Running demo with known compounds:\n")
        smiles_list = [
            'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',  # Gefitinib
            'C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1',        # Erlotinib
            'CC(=O)Oc1ccccc1C(=O)O',                              # Aspirin (control)
        ]

    predict_smiles(smiles_list, args.model_dir)


if __name__ == '__main__':
    main()
