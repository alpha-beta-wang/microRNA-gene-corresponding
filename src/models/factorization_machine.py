import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class FactorizationMachineClassifier:
    """Binary classifier using Factorization Machine with SGD.

    y(x) = w0 + sum(w_i * x_i) + sum_i sum_j>i (v_i · v_j) * x_i * x_j

    References:
        Rendle, S. "Factorization Machines" (2010)
    """

    def __init__(
        self,
        n_factors: int = 8,
        learning_rate: float = 0.01,
        epochs: int = 200,
        batch_size: int = 64,
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
        # Linear term
        linear = self.w0 + X @ self.w  # (n,)

        # Pairwise interactions via O(kn) trick
        # sum_f [(sum_i v_if * x_i)^2 - sum_i (v_if * x_i)^2]
        X_V = X @ self.V  # (n, k)
        X_V_sq = (X ** 2) @ (self.V ** 2)  # (n, k)
        interactions = 0.5 * np.sum(X_V ** 2 - X_V_sq, axis=1)  # (n,)

        return linear + interactions

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FactorizationMachineClassifier":
        if isinstance(y, pd.Series):
            y = y.values
        n, p = X.shape
        self.w0 = 0.0
        self.w = self.rng.normal(0, 0.01, p)
        self.V = self.rng.normal(0, 0.01, (p, self.n_factors))

        n_samples = n
        for epoch in range(self.epochs):
            # Shuffle
            idx = self.rng.permutation(n_samples)
            X_s, y_s = X[idx], y[idx]

            epoch_loss = 0.0
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_b = X_s[start:end]
                y_b = y_s[start:end]
                m = X_b.shape[0]

                # Forward
                raw = self._predict_raw(X_b)
                proba = _sigmoid(raw)
                loss = -(y_b * np.log(proba + 1e-15) + (1 - y_b) * np.log(1 - proba + 1e-15))
                epoch_loss += loss.sum()

                # Gradient
                diff = proba - y_b  # (m,)

                # w0
                grad_w0 = diff.sum()
                self.w0 -= self.learning_rate * grad_w0

                # w
                grad_w = X_b.T @ diff + self.reg_w * self.w  # (p,)
                self.w -= self.learning_rate * grad_w

                # V
                # sum_f: grad_v_if = x_i * sum_j (diff_j * x_j * v_jf) - reg_v * v_if
                X_V = X_b @ self.V  # (m, k)
                diff_2d = diff[:, np.newaxis]  # (m, 1)
                # v_if_grad_inner = sum_j (diff_j * x_j * v_jf) = x_b^T (diff * X_V)
                # Actually: per sample j: diff_j * sum_f v_if * x_i
                # grad_v_if = x_i * sum_j diff_j * (x_j * v_jf) - reg_v * v_if
                # = x_i * (X_b * diff_2d).T @ (X_b @ V[:,f]) ... messy

                # Simpler: compute per-batch-element then sum
                # grad for V[f,:] = sum over samples: diff_j * x_j * (x @ V) - reg_v * V
                # grad_V = X_b.T @ (diff_2d * X_V) - reg_v * self.V
                # But we need to be careful: (diff_2d * X_V) has shape (m, k)
                grad_V = X_b.T @ (diff_2d * X_V)  # (p, k)
                # Subtract the diagonal: - x_i^2 * v_if
                X_b_sq = X_b ** 2  # (m, p)
                # For each factor f: sum_j diff_j * x_jf^2 * v_if
                # Actually: grad += diff * x_i^2 * v_if ... need to be more careful
                # The full gradient for V:
                # dL/dv_if = sum_j [diff_j * x_ji * (sum_l x_jl * v_lf)] - diff_j * x_ji^2 * v_if + reg_v * v_if
                # = sum_j diff_j * x_ji * sum_l x_jl * v_lf - sum_j diff_j * x_ji^2 * v_if + reg_v * v_if
                # First term: X_b.T @ (diff_2d * (X_b @ self.V)) -> (p, k)
                # Second term: diag(X_b.T @ diff_2d) expanded... actually per factor:
                #   sum_j diff_j * x_ji^2 * v_if = v_if * sum_j diff_j * x_ji^2
                #   = self.V[i,f] * (X_b_sq.T @ diff)[i]
                sq_grad = (X_b_sq.T @ diff)[:, np.newaxis] * self.V  # (p, k)

                grad_V = grad_V - sq_grad + self.reg_v * self.V
                self.V -= self.learning_rate * grad_V

            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                train_pred = self.predict_proba(X)
                train_acc = ((train_pred > 0.5) == y).mean()
                print(f"  [fm] epoch {epoch + 1}/{self.epochs} loss={epoch_loss / m:.4f} acc={train_acc:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._predict_raw(X)
        pos = _sigmoid(raw)
        return np.column_stack([1 - pos, pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
