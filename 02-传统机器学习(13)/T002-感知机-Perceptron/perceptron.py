"""T002 · Perceptron Learning Algorithm (PLA)."""

from __future__ import annotations

import numpy as np


class Perceptron:
    """经典感知机（PLA）。标签约定为 {-1, +1}。"""

    def __init__(self, lr: float = 1.0, max_epochs: int = 100):
        self.lr = lr
        self.max_epochs = max_epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.errors_history: list[int] = []
        self.n_updates: int = 0
        self.n_epochs_ran: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if set(np.unique(y)) - {-1.0, 1.0}:
            raise ValueError("标签必须为 {-1, +1}")

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.errors_history = []
        self.n_updates = 0

        for epoch in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = np.sign(np.dot(xi, self.w) + self.b)
                if pred == 0:
                    pred = -1.0
                if pred != yi:
                    update = self.lr * yi
                    self.w += update * xi
                    self.b += update
                    errors += 1
                    self.n_updates += 1
            self.errors_history.append(errors)
            self.n_epochs_ran = epoch + 1
            if errors == 0:
                break
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        preds = np.sign(scores)
        preds[preds == 0] = -1.0
        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        return float((self.predict(X) == y).mean())
