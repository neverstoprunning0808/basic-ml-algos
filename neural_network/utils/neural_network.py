from typing import Tuple

import numpy as np


def signmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def one_hot_encoding_label(y: np.ndarray | list, num_labels: int) -> np.ndarray:
    y = np.array(y)
    ohe_label = np.zeros((y.shape[0], num_labels))
    ohe_label[np.arange(y.shape[0]), y] = 1
    return ohe_label


class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_class, random_seed=123):
        generator = np.random.RandomState(random_seed)

        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_class = num_class

        # hidden layer:
        # z = x.W.T -> W.T = (num_features, num_hidden) -> W = (num_hidden, num_featuers)
        self.weight_hidden = generator.normal(
            loc=0.0, scale=0.1, size=(num_hidden, num_features)
        )
        self.bias_hidden = np.zeros(num_hidden)

        # output layer:
        self.weight_output = generator.normal(
            loc=0.0, scale=0.1, size=(num_class, num_hidden)
        )
        self.bias_output = np.zeros(num_class)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # hidden
        z_hidden = x.dot(self.weight_hidden.T) + self.bias_hidden
        a_hidden = signmoid(z_hidden)

        # output
        z_out = a_hidden.dot(self.weight_output.T) + self.bias_output
        a_output = signmoid(z_out)

        return a_hidden, a_output

    def backward(
        self, x: np.ndarray, a_h: np.ndarray, a_out: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # one hot encoding labels
        y_onehot = one_hot_encoding_label(y, self.num_class)

        #######################################################################################
        # Part 1:
        # dLoss/dW_output = dLoss/dA_output * dA_output/dZ_output * dZ_output/dW_output
        # dLoss/dB_output = dLoss/dA_output * dA_output/dZ_output * dZ_output/dB_output
        #######################################################################################

        # assuming that the Loss function is MSE like Adaline
        dLoss__dA_output = 2 * (a_out - y_onehot) / y.shape[0]  # [n_samples, num_class]

        # derivative of sigmoid function = a*(1-a)
        dA_output__dZ_output = a_out * (1 - a_out)  # [n_samples, num_class]

        # z_output = a_h . W_output.T
        dZ_output__dW_output = a_h  # a_h = X.W.T = [num_samples, num_hidden]

        # put them together:
        dLoss__dW_output = np.dot(
            (dLoss__dA_output * dA_output__dZ_output).T, dZ_output__dW_output
        )  # [n_samples, num_hidden]
        dLoss__dB_output = np.sum(
            (dLoss__dA_output * dA_output__dZ_output), axis=0
        )  # [num_class]

        ###########################################################################################################################
        # Part 2:
        # dLoss/dW_hidden= dLoss/dA_output * dA_output/dZ_output * dZ_output/dA_hidden * dA_hidden/dZ_hidden * dZ_hidden/dW_hidden
        # dLoss/dW_hidden= dLoss/dA_output * dA_output/dZ_output * dZ_output/dA_hidden * dA_hidden/dZ_hidden * dZ_hidden/dB_hidden
        ###########################################################################################################################

        # z_output = a_hidden . W.T + bias
        dZ_output__dA_hidden = self.weight_output  # [num_class, num_hidden]

        # sigmoid derivative
        dA_hidden__dZ_hidden = a_h * (1 - a_h)  # [num_samples, num_hidden]

        # z_hidden = x. W.T + bias
        dZ_hidden__dW_hidden = x  # [num_samples, num_features]

        # put them together
        dLoss__dW_hidden = np.dot(
            (
                np.dot((dLoss__dA_output * dA_output__dZ_output), dZ_output__dA_hidden)
                * dA_hidden__dZ_hidden
            ).T,
            dZ_hidden__dW_hidden,
        )
        dLoss__dB_hidden = np.sum(
            np.dot((dLoss__dA_output * dA_output__dZ_output), dZ_output__dA_hidden)
            * dA_hidden__dZ_hidden,
            axis=0,
        )

        return (dLoss__dW_output, dLoss__dB_output, dLoss__dW_hidden, dLoss__dB_hidden)


# if __name__ == "__main__":
#     y = [0, 1, 2]
#     num_labels = 3
#     print(one_hot_encoding_label(y, num_labels))

#     model = NeuralNetMLP(num_features=2, num_hidden=2, num_class=3, random_seed=1)

#     X = np.random.randn(3, 2)
#     print(X)

#     a, b = model.forward(X)
#     print(a)
#     print(b)
