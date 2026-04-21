import argparse
from typing import Literal, Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_iris_data(
    num_class: Literal["binary", "multi"] = "binary",
    random_state: int = 1,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iris = load_iris()
    features = iris.data
    labels = iris.target

    scaler = StandardScaler()

    if num_class == "binary":
        X = features[(labels == 0) | (labels == 1)]
        y = labels[(labels == 0) | (labels == 1)]
    else:
        X = features
        y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_class",
        type=str,
        choices=["binary", "multi"],
        default="binary",
        help="Choose binary or multi",
    )

    args = parser.parse_args()

    X_train, X_test, y_train, y_test = generate_iris_data(num_class=args.num_class)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
