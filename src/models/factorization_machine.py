import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    eps = 1e-15
    return -(y_true * np.log(np.clip(y_pred, eps, 1 - eps))
             + (1 - y_true) * np.log(np.clip(1 - y_pred, eps, 1 - eps)))


class FactorizationMachineClassifier:
    """Binary classifier using Factorization Machine with SGD.

    y(x) = w0 + sum(w_i * x_i) + sum_i sum_j>i (v_i · v_j) * x_i * x_j

    References:
        Rendle, S. "Factorization Machines" (2010)
    """

    def __init__(
        self,
        n_factors: int = 8,
        learning_rate: float = 0.001,
        epochs: int = 500,
        batch_size: int = 128,
        reg_w: float = 0.001,
        reg_v: float = 0.001,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(random_state)

        self.w0: float = 0.0
        self.w: np.ndarray | None = None
        self.V: np.ndarray | None = None

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Linear + pairwise interaction terms (no sigmoid)."""
        linear = self.w0 + X @ self.w
        # O(kn) interaction trick with overflow protection
        X_V = X @ self.V
        X_V = np.clip(X_V, -50, 50)
        X_V_sq = (X.astype(np.float64) ** 2) @ (self.V.astype(np.float64) ** 2)
        interactions = 0.5 * np.sum(X_V ** 2 - X_V_sq, axis=1)
        return np.clip(linear + interactions, -30, 30)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FactorizationMachineClassifier":
        if isinstance(y, pd.Series):
            y = y.values
        n, p = X.shape
        self.w0 = 0.0
        self.w = self.rng.normal(0, 0.01, p).astype(np.float64)
        self.V = self.rng.normal(0, 0.01, (p, self.n_factors)).astype(np.float64)

        for epoch in range(self.epochs):
            idx = self.rng.permutation(n)
            X_s, y_s = X[idx], y[idx]

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                X_b = X_s[start:end]
                y_b = y_s[start:end]
                m = X_b.shape[0]
                n_batches += 1

                raw = self._predict_raw(X_b)
                proba = _sigmoid(raw)
                epoch_loss += _log_loss(y_b, proba).sum()

                diff = proba - y_b  # (m,)

                # w0
                self.w0 -= self.learning_rate * diff.sum()

                # w
                grad_w = X_b.T @ diff + self.reg_w * self.w
                self.w -= self.learning_rate * grad_w

                # V
                X_V = X_b @ self.V
                diff_2d = diff[:, np.newaxis]
                grad_V = X_b.T @ (diff_2d * X_V)
                X_b_sq = X_b.astype(np.float64) ** 2
                sq_grad = (X_b_sq.T @ diff)[:, np.newaxis] * self.V
                grad_V = grad_V - sq_grad + self.reg_v * self.V
                # Global gradient norm clipping
                grad_norm = np.sqrt((grad_V ** 2).sum())
                if grad_norm > 10.0:
                    grad_V *= 10.0 / grad_norm
                self.V -= self.learning_rate * grad_V

            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                train_pred = self.predict_proba(X)
                train_acc = ((train_pred > 0.5) == y).mean()
                print(f"  [fm] epoch {epoch + 1}/{self.epochs} "
                      f"loss={epoch_loss / n_batches:.4f} acc={train_acc:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._predict_raw(X)
        pos = _sigmoid(raw)
        return np.column_stack([1 - pos, pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
