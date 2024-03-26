"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        #1 Compute the linear combination of inputs and weights
            # z = input * weights + bias
        #2 Compute the predicted probabilities using the sigmoid function
            # y_labels = sigmoid(z)
        #3 Compute the gradient of the loss function with respect to weights
            # gradient_weights = 1/N * X.T * (y_labels - y_train_labels )
        #4 Compute the gradient of the loss function with respect to bias
            # gradient_bias = 1/N * sum(y_labels - y_train_labels )
        #5 Update the weights using the gradient descent rule
            # weights -= learning_rate * gradient_weights
        #6 Update the bias using the gradient descent rule
            # bias -= learning_rate * gradient_bias

        #1
        #weights need to be same dimension as data, bias starts at 0
        N, D = X_train.shape
        self.w = np.zeros(D)
        self.bias = 0

        # to improve classification, we can try and replace all 0 values with -1

        for i in range(self.epochs):
            z = np.dot(X_train, self.w) + self.bias
            y_pred = self.sigmoid(z)
            # print('y_pred shape', y_pred.shape)
            # print('y_train shape', y_train.shape)
            dw = (1/N) * np.dot(X_train.T, (y_pred - y_train))
            db = (1/N) * sum(y_pred - y_train)
            self.w = self.w - self.lr * dw
            print('w', self.w)
            self.bias = self.bias - self.lr * db

        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        # Initialize an empty array to store predicted labels
        # for each example in X_test:
        #     Compute the linear combination of inputs and weights
        #     Compute the predicted probability using the sigmoid function
        #     If the predicted probability is greater than 0.5:
        #         Append 1 to the predicted labels array
        #     Else:
        #         Append 0 to the predicted labels array
        # Return the predicted labels array
        N, D = X_test.shape
        y_pred = np.zeros(D)

        # we can now use the ideal updated weights and biases
        # it's like training but no updating weights, same code
        z = np.dot(X_test, self.w) + self.bias
        y_pred = self.sigmoid(z)
                
        return (y_pred > self.threshold).astype(int)
