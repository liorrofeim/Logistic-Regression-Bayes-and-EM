import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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
        # normalize the data
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        # initialize theta
        self.theta = np.random.rand(X.shape[1] + 1)
        # add x0 = 1 to each example
        X = np.c_[np.ones((X.shape[0], 1)), X]

        for i in range(self.n_iter):
            # calculate the cost function for the initial theta
            z = np.dot(X, self.theta)
            h = 1.0 / (1.0 + np.exp(-z))
            # calculate the gradient
            gradient = np.dot(X.T, (h - y))
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
        # normalize the data
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
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
    p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data - mu) / sigma) ** 2)

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
        # Initialize mixture weights to uniform
        self.weights = np.full(self.k, 1.0 / self.k)

        # Initialize means randomly sampled from the data
        random_indices = np.random.choice(len(data), self.k, replace=False)
        self.mus = data[random_indices]

        # Initialize standard deviations to the standard deviation of the data
        self.sigmas = np.full((self.k, data.shape[1]), np.std(data, axis=0))

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

        self.responsibilities = np.zeros((data.shape[0], self.k))

        for i in range(self.k):
            for j in range(data.shape[1]):
                self.responsibilities[:, i] += self.weights[i] * norm_pdf(
                    data[:, j], self.mus[i, j], self.sigmas[i, j]
                )

        # Normalize responsibilities so they sum to 1
        self.responsibilities /= self.responsibilities.sum(axis=1, keepdims=True)
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
            resp = self.responsibilities[:, i]
            total_resp = resp.sum()

            for j in range(data.shape[1]):
                self.mus[i, j] = (data[:, j] * resp).sum() / total_resp
                self.sigmas[i, j] = np.sqrt(
                    ((data[:, j] - self.mus[i, j]) ** 2 * resp).sum() / total_resp
                )

            self.weights[i] = total_resp / len(data)
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
        self.init_params(data)

        for _ in range(self.n_iter):
            prev_cost = self.costs
            self.expectation(data)
            self.maximization(data)

            # Calculate cost
            self.costs = 0
            for j in range(data.shape[1]):
                self.costs += np.sum(
                    np.log(
                        np.sum(
                            self.weights
                            * norm_pdf(
                                data[:, j, np.newaxis],
                                self.mus[:, j],
                                self.sigmas[:, j],
                            ),
                            axis=1,
                        )
                    )
                )

            # Check for convergence
            if prev_cost is not None and abs(self.costs - prev_cost) < self.eps:
                break

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
    pdf = np.zeros(data.shape[0])

    for i in range(weights.shape[0]):  # for each component
        component_pdf = np.ones(data.shape[0])
        for j in range(data.shape[1]):  # for each dimension
            component_pdf *= norm_pdf(data[:, j], mus[i, j], sigmas[i, j])
        pdf += weights[i] * component_pdf

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
        self.classes = None
        self.em_models = None

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
        # Compute the prior probabilities and fit GMMs to each class
        self.classes = np.unique(y)
        self.em_models = {
            c: EM(k=self.k, random_state=self.random_state) for c in self.classes
        }
        self.prior = {c: np.mean(y == c) for c in self.classes}

        for c in self.classes:
            self.em_models[c].fit(X[y == c])
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
        preds = []
        for i in range(X.shape[0]):
            probs = {
                c: self.prior[c]
                * gmm_pdf(X[i].reshape(1, -1), *self.em_models[c].get_dist_params())
                for c in self.classes
            }
            preds.append(max(probs, key=probs.get))
        preds = np.array(preds)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


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
    # Initialize the models
    lor_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    bayes_model = NaiveBayesGaussian(k=k)

    # Train the models
    lor_model.fit(x_train, y_train)
    bayes_model.fit(x_train, y_train)

    # Compute predictions for train and test datasets
    lor_train_preds = lor_model.predict(x_train)
    lor_test_preds = lor_model.predict(x_test)
    bayes_train_preds = bayes_model.predict(x_train)
    bayes_test_preds = bayes_model.predict(x_test)

    # Compute accuracies
    lor_train_acc = accuracy(y_train, lor_train_preds)
    lor_test_acc = accuracy(y_test, lor_test_preds)
    bayes_train_acc = accuracy(y_train, bayes_train_preds)
    bayes_test_acc = accuracy(y_test, bayes_test_preds)

    # Plot the decision boundaries for the Logistic Regression model
    plot_decision_regions(
        x_train,
        y_train,
        lor_model,
        resolution=0.01,
        title="Logistic Regression Decision Boundaries",
    )

    # Plot decision boundaries for Naive Bayes GMM
    plot_decision_regions(
        x_train,
        y_train,
        bayes_model,
        resolution=0.05,
        title="Naive Bayes GMM Decision Boundaries",
    )
    plt.plot(np.arange(len(lor_model.Js)), lor_model.Js)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss as a function of iterations")
    plt.show()

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

    np.random.seed(0)

    # For dataset_a where Naive Bayes works better, we generate data from multivariate gaussians
    # where the covariance matrix is a diagonal matrix. This leads to the features being independent.
    mean_a1 = [3, 6, 3]
    cov_a1 = np.diag([1, 1, 1])
    mean_a2 = [3, 0, 2]
    cov_a2 = np.diag([1, 1, 1])
    mean_a3 = [-2, 3, 3]
    cov_a3 = np.diag([1, 1, 1])
    mean_a4 = [8, 3, 3]
    cov_a4 = np.diag([1, 1, 1])

    dataset_a1 = multivariate_normal.rvs(mean=mean_a1, cov=cov_a1, size=250)
    dataset_a2 = multivariate_normal.rvs(mean=mean_a2, cov=cov_a2, size=250)
    dataset_a3 = multivariate_normal.rvs(mean=mean_a3, cov=cov_a3, size=250)
    dataset_a4 = multivariate_normal.rvs(mean=mean_a4, cov=cov_a4, size=250)

    dataset_a_features = np.vstack((dataset_a1, dataset_a2, dataset_a3, dataset_a4))
    dataset_a_labels = np.hstack((np.zeros(500), np.ones(500)))

    # For dataset_b where Logistic Regression works better, we generate data from multivariate gaussians
    # where the covariance matrix is not a diagonal matrix, introducing correlations between the features.
    mean_b1 = [0, 0, 0]
    cov_b1 = [[0.9, 2, 0.9], [0.9, 2, 0.9], [0.9, 0.9, 2]]

    mean_b2 = [0, 0, 1]
    cov_b2 = [[0.9, 2, 0.9], [0.9, 2, 0.9], [0.9, 0.9, 2]]

    mean_b3 = [0, 0, 0]
    cov_b3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    mean_b4 = [5, 5, 5]
    cov_b4 = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]

    dataset_b1 = multivariate_normal.rvs(mean=mean_b1, cov=cov_b1, size=500)
    dataset_b2 = multivariate_normal.rvs(mean=mean_b2, cov=cov_b2, size=500)
    # dataset_b3 = multivariate_normal.rvs(mean=mean_b3, cov=cov_b3, size=250)
    # dataset_b4 = multivariate_normal.rvs(mean=mean_b4, cov=cov_b4, size=250)

    dataset_b_features = np.vstack((dataset_b1, dataset_b2))
    dataset_b_labels = np.hstack((np.zeros(500), np.ones(500)))

    return {
        "dataset_a_features": dataset_a_features,
        "dataset_a_labels": dataset_a_labels,
        "dataset_b_features": dataset_b_features,
        "dataset_b_labels": dataset_b_labels,
    }


def generate_datasets2():
    from scipy.stats import multivariate_normal

    n_samples = 1000

    # Dataset A: Naive Bayes performs better
    mean0 = [0, 0, 0]
    cov0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class0 = multivariate_normal(mean0, cov0).rvs(n_samples // 2)

    mean1 = [2, 2, 2]
    cov1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class1 = multivariate_normal(mean1, cov1).rvs(n_samples // 2)

    dataset_a_features = np.vstack((class0, class1))
    dataset_a_labels = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    # Plotting Dataset A
    plt.figure(figsize=(18, 6))
    for i, (x, y) in enumerate([(0, 1), (0, 2), (1, 2)], start=1):
        plt.subplot(1, 3, i)
        plt.scatter(
            dataset_a_features[: n_samples // 2, x],
            dataset_a_features[: n_samples // 2, y],
            color="b",
        )
        plt.scatter(
            dataset_a_features[n_samples // 2 :, x],
            dataset_a_features[n_samples // 2 :, y],
            color="r",
        )
        plt.xlabel(f"Feature {x+1}")
        plt.ylabel(f"Feature {y+1}")
    plt.suptitle("Dataset A")
    plt.show()

    # Dataset B: Logistic Regression performs better
    mean0 = [0, 0, 0]
    cov0 = np.array([[1, 0.9, 0.9], [0.9, 1, 0.9], [0.9, 0.9, 1]])
    class0 = multivariate_normal(mean0, cov0).rvs(n_samples // 2)

    mean1 = [1, 1, 1]
    cov1 = np.array([[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.2, 1]])
    class1 = multivariate_normal(mean1, cov1).rvs(n_samples // 2)

    dataset_b_features = np.vstack((class0, class1))
    dataset_b_labels = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    # Plotting Dataset B
    plt.figure(figsize=(18, 6))
    for i, (x, y) in enumerate([(0, 1), (0, 2), (1, 2)], start=1):
        plt.subplot(1, 3, i)
        plt.scatter(
            dataset_b_features[: n_samples // 2, x],
            dataset_b_features[: n_samples // 2, y],
            color="b",
        )
        plt.scatter(
            dataset_b_features[n_samples // 2 :, x],
            dataset_b_features[n_samples // 2 :, y],
            color="r",
        )
        plt.xlabel(f"Feature {x+1}")
        plt.ylabel(f"Feature {y+1}")
    plt.suptitle("Dataset B")
    plt.show()

    return {
        "dataset_a_features": dataset_a_features,
        "dataset_a_labels": dataset_a_labels,
        "dataset_b_features": dataset_b_features,
        "dataset_b_labels": dataset_b_labels,
    }


"""--------------------------------------------------------------"""

# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = (".", ".")
    colors = ("blue", "red")
    cmap = ListedColormap(colors[: len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor="black",
        )
    plt.show()


def logistic_vs_naive_acc(
    dataset_a_features, dataset_a_labels, best_eta, best_eps, k=2
):
    # the function get dataset and returns the accuracy of logistic regression and naive bayes
    data_a = np.concatenate(
        (dataset_a_features, dataset_a_labels.reshape(-1, 1)), axis=1
    )
    # Shuffle the data
    np.random.seed(1)

    np.random.shuffle(data_a)
    shuffled_indices = np.random.permutation(len(dataset_a_features))
    shuffled_features_a = dataset_a_features[shuffled_indices]
    shuffled_labels_a = dataset_a_labels[shuffled_indices]

    split_index = int(0.8 * shuffled_features_a.shape[0])

    # Split the shuffled features
    train_features_a = shuffled_features_a[:split_index]
    test_features_a = shuffled_features_a[split_index:]

    # Split the shuffled labels
    train_labels_a = shuffled_labels_a[:split_index]
    test_labels_a = shuffled_labels_a[split_index:]

    lor_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor_model.fit(train_features_a, train_labels_a)

    lor_test_preds = lor_model.predict(test_features_a)
    lor_acc = accuracy(lor_test_preds, test_labels_a)

    bayes_model = NaiveBayesGaussian(k=k)
    bayes_model.fit(train_features_a, train_labels_a)
    bayes_test_preds = bayes_model.predict(test_features_a)
    naive_acc = accuracy(bayes_test_preds, test_labels_a)
    return lor_acc, naive_acc
