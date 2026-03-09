"""
PyTorch neural network for cross-section prediction with MC dropout uncertainty.

Architecture: residual fully-connected network with batch normalization and
Monte Carlo dropout for approximate Bayesian uncertainty quantification.

Reference for MC dropout:
  Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing
  Model Uncertainty in Deep Learning," ICML 2016.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """
    Residual block: Linear -> BN -> ReLU -> Linear -> BN with skip connection.

    Uses pre-activation residual design for stable training.
    """

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class CrossSectionNet(nn.Module):
    """
    Residual neural network for log cross-section prediction.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_dims : list of int
        Sizes of hidden layers. Each entry creates a projection + residual block.
    dropout_rate : float
        Dropout probability (active at both train and inference time for MC dropout).
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: List[int] = (256, 256, 128, 64),
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_dim = n_features

        for out_dim in hidden_dims:
            # Projection layer to change dimensions
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

            # Residual block maintaining dimension
            layers.append(ResidualBlock(out_dim, dropout_rate=dropout_rate))
            in_dim = out_dim

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.output_layer(features).squeeze(-1)

    def enable_dropout(self):
        """Enable dropout layers for MC inference (override eval mode)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


class NeuralNetworkCrossSectionModel:
    """
    PyTorch neural network wrapper for cross-section prediction.

    Uses MC dropout to sample approximate posterior predictive distribution
    at inference time.

    Parameters
    ----------
    n_features : int
        Input feature dimensionality.
    hidden_dims : list of int
        Hidden layer dimensions.
    dropout_rate : float
        Dropout probability.
    learning_rate : float
        Adam optimizer learning rate.
    weight_decay : float
        L2 regularization strength.
    batch_size : int
        Mini-batch size for training.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience in epochs.
    mc_samples : int
        Number of MC dropout forward passes for uncertainty estimation.
    device : str or None
        "cuda", "cpu", or None (auto-detect).
    feature_names : list of str, optional
        Feature names for interpretability.
    """

    def __init__(
        self,
        n_features: int = 22,
        hidden_dims: Tuple[int, ...] = (256, 256, 128, 64),
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 20,
        mc_samples: int = 100,
        device: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.n_features = n_features
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.mc_samples = mc_samples
        self.feature_names = feature_names

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model: Optional[CrossSectionNet] = None
        self._fitted = False
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []

    def _build_model(self) -> CrossSectionNet:
        return CrossSectionNet(
            n_features=self.n_features,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

    def _make_dataloader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "NeuralNetworkCrossSectionModel":
        """
        Train the neural network.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data.
        X_val, y_val : np.ndarray, optional
            Validation data for early stopping.

        Returns
        -------
        self
        """
        logger.info(
            "Training neural network on %d samples (device=%s)", len(X_train), self.device
        )

        self._model = self._build_model()
        optimizer = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        criterion = nn.MSELoss()

        train_loader = self._make_dataloader(X_train, y_train, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training phase
            self._model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self._model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(X_train)
            self._train_losses.append(train_loss)

            # Validation phase
            if X_val is not None:
                val_loss = self._compute_loss(X_val, y_val, criterion)
                self._val_losses.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 20 == 0:
                    logger.info(
                        "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f",
                        epoch, self.max_epochs, train_loss, val_loss,
                    )

                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.patience)
                    break
            else:
                if epoch % 20 == 0:
                    logger.info(
                        "Epoch %d/%d: train_loss=%.4f", epoch, self.max_epochs, train_loss
                    )

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
            logger.info("Restored best model with val_loss=%.4f", best_val_loss)

        self._fitted = True
        return self

    def _compute_loss(
        self, X: np.ndarray, y: np.ndarray, criterion: nn.Module
    ) -> float:
        """Compute mean loss on a dataset."""
        self._model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = self._model(X_t)
            loss = criterion(y_pred, y_t).item()
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return mean prediction (mean over MC dropout samples)."""
        result = self.predict_with_uncertainty(X)
        return result["mean"]

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Predict with MC dropout uncertainty estimation.

        Runs self.mc_samples forward passes with dropout active and
        computes statistics over the sample distribution.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        dict with keys: mean, std, lower_68, upper_68, lower_95, upper_95,
                        q05, q16, median, q84, q95.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_with_uncertainty().")

        self._model.eval()
        self._model.enable_dropout()

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        sample_preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds = self._model(X_t).cpu().numpy()
                sample_preds.append(preds)

        samples = np.stack(sample_preds, axis=0)  # (mc_samples, n_test)

        mean = samples.mean(axis=0)
        std = samples.std(axis=0)

        return {
            "mean": mean,
            "std": std,
            "lower_68": mean - std,
            "upper_68": mean + std,
            "lower_95": mean - 2 * std,
            "upper_95": mean + 2 * std,
            "q05": np.percentile(samples, 5, axis=0),
            "q16": np.percentile(samples, 16, axis=0),
            "median": np.median(samples, axis=0),
            "q84": np.percentile(samples, 84, axis=0),
            "q95": np.percentile(samples, 95, axis=0),
        }

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import mean_squared_error, r2_score

        preds = self.predict_with_uncertainty(X_test)
        y_pred = preds["mean"]

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(np.mean(np.abs(y_test - y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        coverage_68 = float(np.mean(
            (y_test >= preds["lower_68"]) & (y_test <= preds["upper_68"])
        ))
        coverage_95 = float(np.mean(
            (y_test >= preds["lower_95"]) & (y_test <= preds["upper_95"])
        ))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "picp_68": coverage_68,
            "picp_95": coverage_95,
            "mpiw_68": float(np.mean(preds["upper_68"] - preds["lower_68"])),
            "mpiw_95": float(np.mean(preds["upper_95"] - preds["lower_95"])),
            "mean_epistemic_std": float(np.mean(preds["std"])),
        }

    def save(self, path: str) -> None:
        """Save model weights and configuration."""
        save_path = Path(path)
        # Save network weights
        torch.save(self._model.state_dict(), str(save_path) + ".pt")
        # Save configuration
        config = {
            "n_features": self.n_features,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "mc_samples": self.mc_samples,
            "feature_names": self.feature_names,
        }
        with open(str(save_path) + ".pkl", "wb") as f:
            pickle.dump(config, f)
        logger.info("Saved neural network to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "NeuralNetworkCrossSectionModel":
        """Load model from disk."""
        with open(str(path) + ".pkl", "rb") as f:
            config = pickle.load(f)

        instance = cls(device=device, **config)
        instance._model = instance._build_model()
        weights = torch.load(str(path) + ".pt", map_location=instance.device)
        instance._model.load_state_dict(weights)
        instance._fitted = True
        return instance
