"""Neural network and Random Forest models for binding affinity prediction."""

import torch
import torch.nn as nn
import numpy as np


class BindingAffinityNet(nn.Module):
    """Feedforward network for predicting pIC50 from molecular fingerprints."""

    def __init__(self, input_dim: int = 2054, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        # Kaiming init works better than default for ReLU networks
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class RandomForestBaseline:
    """
    Random Forest regressor — simple baseline, often surprisingly competitive.
    Always train this alongside the NN for comparison.
    """

    def __init__(self, n_estimators: int = 200, n_jobs: int = -1):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features='sqrt',
            min_samples_leaf=2,
            n_jobs=n_jobs,
            random_state=42,
        )
        self.trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        print(f"Training Random Forest ({self.model.n_estimators} trees)...")
        self.model.fit(X, y)
        self.trained = True
        print("  Done!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained yet")
        return self.model.predict(X)

    def feature_importances(self, top_k: int = 20) -> np.ndarray:
        if not self.trained:
            return None
        return np.argsort(-self.model.feature_importances_)[:top_k]


class MoleculeDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper for fingerprint feature arrays."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


if __name__ == '__main__':
    model = BindingAffinityNet(input_dim=2054)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    batch = torch.randn(8, 2054)
    out = model(batch)
    print(f"Output shape: {tuple(out.shape)}")
