from __future__ import annotations

import numpy as np


class LogisticRegression:
    def __init__(self, lr: float = 0.01, n_iter: int = 100, random_state=8) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def linear(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.w_) + self.b_

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        random_generator = np.random.RandomState(self.random_state)
        self.w_ = random_generator.normal(size=X.shape[1])
        self.b_ = 0.0
        self.losses_ = []

        # in the notebook 04, the update is: -(y-a)x -> gd: +(y-a)x
        for _ in range(self.n_iter):
            z = self.linear(X)
            eps = 1e-10
            a = np.clip(
                self.sigmoid(z), eps, 1 - eps
            )  # a is a soft proba, and we want it in (0, 1) to prevent log(0) and log(1)
            error_term = y - a
            self.w_ += self.lr * X.T.dot(error_term) / X.shape[0]
            self.b_ += self.lr * error_term.mean()
            # caculate loss function: NLL
            loss = (-y.dot(np.log(a)) - (1 - y).dot(np.log(1 - a))) / X.shape[0]
            self.losses_.append(loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.sigmoid(self.linear(X)) >= 0.5, 1, 0)


# test case
# if __name__ == "__main__":
#     X = np.array([[1.1, 1.2], [2.1, 2.2], [1.3, 1.2], [2.3, 2.4]])
#     y = np.array([1, 0, 1, 0])

#     model = LogisticRegression()
#     model.fit(X, y)
#     preds = model.predict(X)
#     print(preds)
#     print(model.losses_[-1])
