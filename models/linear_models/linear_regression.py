import numpy as np


class LinearRegression:
    def __init__(self, lr=0.1, epochs=200, batch_size=None, shuffle=True, seed=42) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = int(seed)

        self.w = None
        self.b = None
        self.loss_history = []
        self._LOG_EVERY = 10
        
    @property
    def coef_(self):
        return self.w

    @property
    def intercept_(self):
        return self.b

    def _init_params(self, d):
        self.w = np.random.randn(d)
        self.b = 0.0

    def predict(self, X):
        assert self.w is not None and self.b is not None
        return X @ self.w + self.b

    def _compute_grads(self, X, y):
        n = X.shape[0]
        y_pred = X @ self.w + self.b
        r = y_pred - y
        grad_w = (2.0 / n) * (X.T @ r)
        grad_b = (2.0 / n) * np.sum(r)
        return grad_w, float(grad_b)

    def _compute_loss(self, X, y):
        y_pred = self.predict(X)
        r = y_pred - y
        mse = (1.0 / X.shape[0]) * np.sum(r**2)
        return mse

    def _make_batches(self, n):
        np.random.seed(self.seed)
        idx = np.arange(n)
        if self.shuffle:
            np.random.shuffle(idx)
        if self.batch_size is None or self.batch_size >= n:
            yield idx
            return
        for start in range(0, n, self.batch_size):
            yield idx[start : start + self.batch_size]

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        self._init_params(d)
        self.loss_history = []

        for epoch in range(1, self.epochs + 1):
            for b in self._make_batches(n):
                Xb = X[b]
                yb = y[b]
                grad_w, grad_b = self._compute_grads(Xb, yb)
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

            train_loss = self._compute_loss(X, y)

            if epoch % self._LOG_EVERY == 0:
                print(f"Epoch {epoch:4d} | train_loss={train_loss:.6f}")

            self.loss_history.append(train_loss)

        return self.loss_history
