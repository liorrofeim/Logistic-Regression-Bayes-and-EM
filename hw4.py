import numpy as np


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # initialize theta
        self.theta = np.random.rand(X.shape[1] + 1)
        # add x0 = 1 to each example
        X = np.c_[np.ones((X.shape[0], 1)), X]

        for i in range(self.n_iter):
            # calculate the cost function for the initial theta
            z = np.dot(X, self.theta)
            h = 1.0 / (1.0 + np.exp(-z))
            # calculate the gradient
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.eta * gradient
            # calculate the cost function for the new theta
            J = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / X.shape[0]

            self.thetas.append(self.theta)
            # stop if we get to convergence
            if len(self.Js) > 0 and abs(self.Js[-1] - J) < self.eps:
                self.Js.append(J)
                break
            self.Js.append(J)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # add x0 = 1 to each example
        X = np.c_[np.ones((X.shape[0], 1)), X]
        # calculate the probability of each example to be 1
        z = np.dot(X, self.theta)
        h = 1.0 / (1.0 + np.exp(-z))
        preds = np.where(h >= 0.5, 1, 0)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # shuffle the data
    shuffled_indices = np.random.permutation(len(X))
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    # split the data into folds
    X_folds = np.array_split(X_shuffled, folds)
    y_folds = np.array_split(y_shuffled, folds)
    # train the model on each fold
    accuracies = []
    for i in range(folds):
        # split the data into train and test
        X_train = np.concatenate(X_folds[:i] + X_folds[i + 1 :])
        y_train = np.concatenate(y_folds[:i] + y_folds[i + 1 :])
        X_test = X_folds[i]
        y_test = y_folds[i]

        # train the model
        algo.fit(X_train, y_train)

        # predict the test set
        y_pred = algo.predict(X_test)
        # calculate the accuracy
        accuracy = np.sum(y_pred == y_test) / y_test.size
        accuracies.append(accuracy)
    # calculate the aggregated metrics
    cv_accuracy = np.mean(accuracies)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mu and sigma for the given x.
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p = np.exp(-0.5 * ((data.flatten() - mu) / sigma) ** 2) / (
        sigma * np.sqrt(2 * np.pi)
    )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        num_data = data.shape[0]
        dim = data.shape[1]

        # Initialize responsibilities randomly
        self.responsibilities = np.random.rand(num_data, self.k)
        self.responsibilities /= self.responsibilities.sum(axis=1, keepdims=True)

        # Initialize weights uniformly
        self.weights = np.ones(self.k) / self.k

        # Initialize mus randomly chosen from data points
        random_indices = np.random.choice(num_data, self.k, replace=False)
        self.mus = data[random_indices]

        # Initialize sigmas to be identity matrices
        self.sigmas = np.array([np.eye(dim) for _ in range(self.k)])

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        # Calculate the responsibilities
        for i in range(self.k):
            # Calculate the pdf
            pdf = norm_pdf(data, self.mus[i], self.sigmas[i])
            # Update responsibilities
            self.responsibilities[:, i] = self.weights[i] * pdf
        # Normalize responsibilities
        self.responsibilities /= np.sum(self.responsibilities, axis=1, keepdims=True)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for i in range(self.k):
            # Compute the effective number of data points assigned to this Gaussian
            weight = self.responsibilities[:, i].sum()

            # Update the weight for this Gaussian
            self.weights[i] = weight / data.shape[0]

            # Update the mean for this Gaussian
            self.mus[i] = (self.responsibilities[:, i, np.newaxis] * data).sum(
                axis=0
            ) / weight

            # Update the standard deviation for this Gaussian
            diff = data - self.mus[i]
            self.sigmas[i] = np.sqrt(
                (self.responsibilities[:, i, np.newaxis] * diff**2).sum(axis=0)
                / weight
            )
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Initialize parameters
        self.init_params(data)

        # For calculating change in cost (log likelihood)
        prev_cost = float("-inf")

        # For keeping track of the costs
        self.costs = []

        for iteration in range(self.n_iter):
            # E step
            self.expectation(data)

            # M step
            self.maximization(data)

            # Calculate cost (log likelihood)
            cost = np.sum(
                self.responsibilities
                * np.log(
                    np.array(
                        [
                            self.weights[i]
                            * norm_pdf(data, self.mus[i], self.sigmas[i])
                            for i in range(self.k)
                        ]
                    ).T
                )
            )

            # Store the cost
            self.costs.append(cost)

            # Check for convergence
            if np.abs(cost - prev_cost) < self.eps:
                break

            prev_cost = cost
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.

    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.

    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    """
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    """

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {
        "lor_train_acc": lor_train_acc,
        "lor_test_acc": lor_test_acc,
        "bayes_train_acc": bayes_train_acc,
        "bayes_test_acc": bayes_test_acc,
    }


def generate_datasets():
    from scipy.stats import multivariate_normal

    """
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    """
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {
        "dataset_a_features": dataset_a_features,
        "dataset_a_labels": dataset_a_labels,
        "dataset_b_features": dataset_b_features,
        "dataset_b_labels": dataset_b_labels,
    }
