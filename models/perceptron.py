"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # 1. Initialize weights vector w and bias term b
        # 2. For each x, compute activation which is basicaly  z = w*x + b
        # 3. Prediction with applying step function
            # If z >= 0, y_pred = 1; else y_pred = -1
        # 4. Compare y_pred and y to get loss
        # 5. Update bias and weights
            # w = w + lr * dotproduct((y - y_pred), x) 
            # b = b + lr * y

        # 1. Initialize the weights vector w with zeros or random small values.
        # 2. For each epoch from 1 to total_epochs:
        #     a. For each training example:
        #         i. Compute the activation value
                    # z = dot_product(w, x_i).
        #         ii. Compute the predicted class label y_pred using the activation value.
        #         iii. If y_pred is not equal to y_i (misclassification):
        #             A. Update the weights vector w using the perceptron update rule:
        #                 w = w + learning_rate * (y_i - y_pred) * x_i
        # 3. Return the trained weights vector w.
        N, D = X_train.shape
        # self.w = np.random.randn(D, self.n_class)
        self.w = np.zeros(D)
        
        for i in range(self.epochs):
            z = np.dot(X_train, self.w)
            y_pred = np.array([1 if val >= 0 else -1 for val in z])
            # print('y pred', y_pred)
            # print('y train', y_train)
            if np.any(y_pred != y_train):
                self.w = self.w - self.lr * np.dot((y_train - y_pred), X_train)

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
        N, D = X_test.shape
        y_pred = np.zeros(N)

        # we can now use the ideal updated weights and biases
        # it's like training but no updating weights, same code
        z = np.dot(X_test, self.w)
        y_pred = np.array([1 if val >= 0 else -1 for val in z])
                
        return y_pred
