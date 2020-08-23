import numpy as np
from numpy.random import shuffle


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def compute_accuracy_score(y_true, y_pred, normalize=True):
    """Accuracy classification score for that model.

    Parameters:
    -----------
    y_true: 1d array
        The ground-truth array.
    y_pred: 1d array
        Predicted values for that arrangement
    normalize: bool, optional (default=True)
        If True returns the fraction of correcly classified samples.
        Returns the number of classifications samples otherwise.
    """
    score = y_true == y_pred

    if normalize:
        return score.sum()/score.shape[0]
    return score.sum()


def compute_cross_val_score(X, y, estimator=None, cv=5):
    """Generate evaluation scores by using cross-validation

    Parameters
    ----------
    estimator
    X
    y
    cv
    
    Returns
    -------
    scores: array
        The scores for each run of cross-validation


    Algorithm Scratch
    -----------------
    Steps to perform the scores computation

    Concatenate X and y: new matrix 'data' will have length 'm'
    Shuffle 'data' randomly    
    Split 'data' into 'k' sets (k = cv) in order to perform cross validation.
    Each set should have length m/k.

    For each unique group:
        1) Take the group as a hold out or test dataset
        2) Take the remaining groups as a training set
        3) Fit a model on the training set and evaluate it on the test set.
        4) Retain the evaluation score and discard the model

    Summarize the skill of the model using the sample of model evaluation scores
    """
    y = y.reshape((y.shape[0], 1))
    data = np.concatenate((X, y), axis=1)
    
    shuffle(data)

    m = data.shape[0]
    k, r = np.divmod(m, cv)
    k, r = int(k), int(r)

    folds = []

    for i in range(cv):
        if (i+1)*k < (m-r):
            folds.append(data[(i*k):(i+1)*k])
        else:
            folds.append(data[(i*k):((i+1)*k)+r])
    
    scores = []

    for i, fold in enumerate(folds):
        training_set = [fold for j, fold in enumerate(folds) if i!=j]
        training_set = [item for sublist in training_set for item in sublist]
        training_set = np.matrix(training_set)
        training_set = np.squeeze(np.asarray(training_set))
        
        X_train, y_train = training_set[:, :-1], training_set[:, -1]
        X_val, y_val = fold[:, :-1], fold[:, -1]
                
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)
        score = float(compute_accuracy_score(y_val, y_pred, normalize=True))
        scores.append(score) # val is the current i - current validation set index
    
    return scores
