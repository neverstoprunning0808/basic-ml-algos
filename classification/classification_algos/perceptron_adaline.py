# The Perceptron and Adaline Algos:

# import libraries
from typing import Literal

import numpy as np


class Perceptron:
    def __init__(
        self,
        n_iter: int = 50,
        lr: float = 0.01,
        model: Literal["perceptron", "adaline"] = "perceptron",
        random_state: int = 1,
    ):
        self.n_iter = n_iter
        self.lr = lr
        self.model = model.lower()
        self.random_state = random_state

        if self.model not in ["perceptron", "adaline"]:
            raise ValueError(
                f"Invalid model name {model}. Expected: 'perceptron' or 'adaline'."
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        random_generation = np.random.RandomState(self.random_state)
        self.w_ = random_generation.normal(size=X.shape[1])
        self.b_ = 0.0

        self.errors_ = []
        self.mse_ = []
        for _ in range(self.n_iter):
            errors = 0.0
            # calculate error
            if self.model == "perceptron":
                for input, target in zip(X, y):  # stochastic update
                    error = target - self.predict(input)

                    # update weights and bias
                    self.w_ += self.lr * error * input
                    self.b_ += self.lr * error
                    # calculate errors
                    errors += error != 0

                self.errors_.append(errors)

            else:  # batch update for adaline
                error = y - self.linear_predict(X)

                # update using gd for mse loss: -(-2 * error * derivate(error)
                self.w_ += (
                    self.lr * 2.0 * (X.T.dot(error) / X.shape[0])
                )  # derivative(error/w) = X
                self.b_ += self.lr * 2.0 * error.mean()  # derivateive(error/b) = 1
                # caclualte mse error
                mse = (error**2).mean()
                self.mse_.append(mse)
                # calculate errors
                self.errors_.append(np.sum(self.predict(X) != y))

    def linear_predict(self, X: np.ndarray) -> float | np.ndarray:
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: np.ndarray) -> int | np.ndarray:
        return np.where(self.linear_predict(X) >= 0.0, 1, 0)
