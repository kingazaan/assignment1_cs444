"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.bias = 0
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.batch_size = 128

    def calc_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss for a mini-batch.

        Parameters:
            X_batch: a numpy array of shape (batch_size, n_features) containing a mini-batch of data
            y_batch: a numpy array of shape (batch_size,) containing training labels

        Returns:
            the gradient with respect to weights w, and bias term
        """
        batch_size = X_batch.shape[0]
        dw = np.zeros_like(self.w)
        db = 0

        for i in range(min(self.batch_size, batch_size)):
            y_pred = np.dot(X_batch[i], self.w) - self.bias
            if y_batch[i] * y_pred < 1:
                dw += -X_batch[i] * y_batch[i]
                db += -y_batch[i]

        dw /= self.batch_size
        db /= self.batch_size

        # Add regularization term to the gradient
        dw += 2 * self.reg_const * self.w

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
        # print('x data', X_train)
        # print('y data', y_train)
        print(X_train)
        N, D = X_train.shape
        self.w = np.zeros(D)
        self.bias = 0
        
        # self.w = np.random.rand(D)

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

                self.w = self.w - self.lr * dw
                self.bias = self.bias - self.lr * db
                
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
        # y = sign (w*x - b)
        print('weights', self.w)
        print('X test', X_test)
        print('bias', self.bias)

        predictions = []
        for x in X_test:
            prediction = np.dot(self.w, x) - self.bias
            if prediction >= 0:
                predictions.append(1)
            else:
                predictions.append(-1)

        # w = self.w.reshape(-1, 1)
        print(predictions)

        return predictions
    
# Notes on old SVM calc gradient

# Incorrect calculation of the hinge loss:
# The hinge loss calculation in your old code is incorrect. It should be based on the predicted values (y_pred) and the true labels (y_train). However, your old code is using y_train[i] directly without involving y_pred.

# Incorrect handling of the bias term:
# In your old code, the bias term (db) is being updated using y_train[i], which is incorrect. The bias term should be updated based on the sum of the hinge loss.

# Old SVM calc_gradient below:
    
    # def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    #     """Calculate gradient of the svm hinge loss.

    #     Inputs have dimension D, there are C classes, and we operate on
    #     mini-batches of N examples.

    #     Parameters:
    #         X_train: a numpy array of shape (N, D) containing a mini-batch
    #             of data
    #         y_train: a numpy array of shape (N,) containing training labels;
    #             y[i] = c means that X[i] has label c, where 0 <= c < C

    #     Returns:
    #         the gradient with respect to weights w; an array of the same shape
    #             as w
    #     """
    #     # TODO: implement me
    #     N, D = X_train.shape
    #     dw = np.zeros_like(self.w)
    #     db = 0

    #     loss = 1 - y_train * (np.dot(X_train, self.w) - self.bias)

    #     for i in range(N):
    #         if loss[i] <= 0:
    #             dw += 2 * self.reg_const * self.w
    #             db += 0
    #         else:
    #             reg_times_w2 = self.reg_const * self.w**2
    #             y_pred = np.dot(X_train[i], y_train[i]) - self.bias
    #             dw += self.reg_const * self.w**2 + 1 - y_train[i] * np.dot(X_train[i], y_train[i]) # - self.bias
    #             db += y_train[i]

    #     return dw, db