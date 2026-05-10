"""
Train a model to predict binding affinity for cancer targets.

Trains both a Random Forest baseline and a feedforward neural network,
then compares them on the test set.

Usage:
    python scripts/train.py                          # all targets combined
    python scripts/train.py --target EGFR            # EGFR only
    python scripts/train.py --target IDH1 --epochs 100
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.molecule_utils import batch_smiles_to_features
from utils.metrics import evaluate_all, print_results
from models.binding_model import BindingAffinityNet, RandomForestBaseline, MoleculeDataset


def load_data(data_dir: Path, target: str = None):
    """Load processed CSV and compute fingerprints."""
    if target:
        csv_path = data_dir / f'{target.lower()}_data.csv'
        if not csv_path.exists():
            csv_path = data_dir / 'bindingdb_cancer.csv'
            print(f"No target-specific file, loading all data and filtering")
    else:
        csv_path = data_dir / 'bindingdb_cancer.csv'

    if not csv_path.exists():
        print(f"ERROR: No processed data found at {data_dir}")
        print("Run: python data/prepare_bindingdb.py first!")
        sys.exit(1)

    print(f"Loading {csv_path.name}...")
    df = pd.read_csv(csv_path)

    if target and 'target_label' in df.columns:
        df = df[df['target_label'] == target.upper()]
        print(f"  Filtered to {target}: {len(df):,} examples")
    else:
        print(f"  Loaded {len(df):,} examples")

    if len(df) == 0:
        print(f"ERROR: No data for target '{target}'")
        sys.exit(1)

    print(f"Computing fingerprints for {len(df):,} molecules...")
    features, valid_indices = batch_smiles_to_features(df['smiles'].tolist())
    labels = df['pIC50'].values[valid_indices]

    print(f"  Valid: {len(valid_indices):,} | Feature shape: {features.shape}")
    print(f"  pIC50 range: {labels.min():.2f} to {labels.max():.2f} | Mean: {labels.mean():.2f}")

    return features, labels


def train_neural_network(X_train, y_train, X_val, y_val,
                         epochs: int = 50, batch_size: int = 256,
                         lr: float = 1e-3, dropout: float = 0.3,
                         save_dir: Path = None):
    """Train the neural network with early stopping based on validation loss."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining Neural Network on {device}")
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

    train_set = MoleculeDataset(X_train, y_train)
    val_set = MoleculeDataset(X_val, y_val)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = BindingAffinityNet(
        input_dim=X_train.shape[1],
        hidden_dims=[1024, 512, 256, 128],
        dropout=dropout,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())
                val_preds.extend(pred.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_pearson = np.corrcoef(y_val, val_preds)[0, 1]

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}  pearson={val_pearson:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': best_state,
            'input_dim': X_train.shape[1],
            'history': history,
        }, save_dir / 'nn_model.pt')
        print(f"  Saved: {save_dir / 'nn_model.pt'}")

    return model, history


@torch.no_grad()
def nn_predict(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i:i+batch_size]).float().to(device)
        preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).parent.parent / 'data' / 'processed')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--compare-baseline', action='store_true', default=True)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent.parent / 'runs')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    target_str = args.target or 'all'
    run_dir = args.output_dir / f'{target_str}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data_dir, args.target)

    # 80/10/10 train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    print(f"\nSplit: train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")

    results = {}

    # Random Forest baseline
    if args.compare_baseline:
        print("\n--- Random Forest Baseline ---")
        rf = RandomForestBaseline(n_estimators=200)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_metrics = evaluate_all(y_test, rf_preds)
        print_results(rf_metrics)
        results['random_forest'] = rf_metrics

        with open(run_dir / 'rf_model.pkl', 'wb') as f:
            pickle.dump(rf, f)

    # Neural Network
    print("\n--- Neural Network ---")
    nn_model, history = train_neural_network(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=run_dir,
    )
    nn_preds = nn_predict(nn_model, X_test)
    nn_metrics = evaluate_all(y_test, nn_preds)
    print("\nNeural Network Test Results:")
    print_results(nn_metrics)
    results['neural_network'] = nn_metrics

    # Side-by-side comparison
    if args.compare_baseline:
        print("\n--- Comparison ---")
        print(f"  {'Metric':<20} {'RF':>8} {'NN':>8}")
        for metric in ['rmse', 'pearson_r', 'spearman_rho', 'roc_auc']:
            rf_v = results['random_forest'][metric]
            nn_v = results['neural_network'][metric]
            winner = 'NN' if (nn_v > rf_v if metric != 'rmse' else nn_v < rf_v) else 'RF'
            print(f"  {metric:<20} {rf_v:>8.4f} {nn_v:>8.4f}  ({winner} wins)")

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {run_dir}")


if __name__ == '__main__':
    main()
