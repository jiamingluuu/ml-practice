import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class LogisticRegression:
    def __init__(
        self,
        lamb=1.0,
        lr=0.1,
        epochs=200,
        batch_size=None,
        shuffle=True,
        seed=42,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = int(seed)

        self.lamb = lamb
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
        rng = np.random.default_rng(seed=self.seed)
        self.w = rng.random((d, ))
        self.b = 0.0

    def predict(self, X):
        assert self.w is not None and self.b is not None
        return sigmoid(X @ self.w + self.b)

    def _compute_grads(self, X, y):
        n = X.shape[0]
        print(y.shape)
        y_pred = self.predict(X)
        r = y_pred - y
        print(X.shape)
        print(r.shape)
        grad_w = (1.0 / n) * X @ r.T + self.lamb * self.w
        grad_b = (1.0 / n) * np.sum(r)
        return grad_w, float(grad_b)

    def _compute_loss(self, X, y):
        # Use probability predictions for loss computation
        # Binary cross-entropy loss (log loss) for logistic regression
        X = np.asarray(X, dtype=np.float64)
        y_pred = sigmoid(X @ self.w + self.b)
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        n = X.shape[0]
        # Binary cross-entropy: -1/n * sum(y*log(y_pred) + (1-y)*log(1-y_pred))
        log_loss = -(1.0 / n) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        reg_loss = 0.5 * self.lamb * np.sum(self.w**2)
        return log_loss + reg_loss

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
