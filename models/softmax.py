"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.bias = 0
        self.batch_size = 128

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        # Compute the scores for each class: scores = X_batch.dot(self.w).
        # Compute the softmax probabilities for each class: probs = softmax(scores).
        # softmax: exp_scores = np.exp(z); return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Compute the gradient of the loss function with respect to the weights:
            # dw = (1/N) * X_batch.T.dot(probs - y_train_one_hot) + 2 * self.reg_const * self.w.
        # Compute the gradient of the loss function with respect to the bias:
            # db = (1/N) * np.sum(probs - y_train_one_hot).
        # Return the gradients dw and db.

        N, D = X_train.shape
        dw = np.zeros_like(self.w)
        db = 0

        y_train_one_hot = np.zeros((N, self.n_class))
        y_train_one_hot[np.arange(N), y_train] = 1

        z = np.dot(X_train, self.w)
        print('scores dot product of train and weights', z)
        probabilities = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True) # aka softmax
        print('probabilities', probabilities)
        # for i in range(N):
        #     probabilities[i, y_train[i]] -= 1
        
        dw = (1/N) * X_train.T.dot(probabilities - y_train_one_hot) + 2 * self.reg_const * self.w
        db = (1/N) * np.sum(probabilities - y_train_one_hot, axis=0)

        # regularization
        dw += self.reg_const * self.w

        return dw, db

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # Initialize the weights self.w to zeros or small random values.
        # Divide the training data into mini-batches.
        # For each epoch:
            # Shuffle the training data.
            # For each mini-batch:
                # Compute the gradients dw and db using calc_gradient.
                # Update the weights: self.w -= self.lr * dw.
                # Update the bias: self.bias -= self.lr * db.

        N, D = X_train.shape
        self.w = np.zeros((D, self.n_class))

        for _ in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, N, self.batch_size):
                # Get mini-batch
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                # print('xbatch', X_batch.shape)
                # print('ybatch', y_batch.shape)
                
                dw, db = self.calc_gradient(X_batch, y_batch)

                self.w -= self.lr * dw
                self.bias -= self.lr * db
        
        # return
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

        z = np.dot(X_test, self.w)
        probabilities = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True) 

        return np.argmax(probabilities, axis=1)

