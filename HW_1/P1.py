"""
Roxanne Krause | 2/5/26 | CS429

TASK
Modify the classes AdalineGD and LogisticRegressionGD 
in the textbook such that the bias
data field b is absorbed by the weight vector w . 
Your program is required to be compatible with the training
programs in the textbook.

REPORT:
Explain how the bias is transformed to an extra weight and why the translated model is equivalent
to the original one.
"""

import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt

"""
code from the text book for the adaline learning model.
below are modifications to absorb the bias into the vectors w and x.
"""
class AdalineGD:
  """ADAptive LInear NEuron classifier.
  Parameters
  ------------
  eta : float
  Learning rate (between 0.0 and 1.0)
  n_iter : int
  Passes over the training dataset.
  random_state : int
  Random number generator seed for random weight initialization.
  Attributes
  -----------
  w_ : 1d-array
  Weights after fitting.
  b_ : Scalar
  Bias unit after fitting.
  losses_ : list
  Mean squared error loss function values in each epoch.
  """
  def __init__(self, eta=0.01, n_iter=50, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

  def fit(self, X, y):
    """ Fit training data.
    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
    Training vectors, where n_examples
    is the number of examples and
    n_features is the number of features.
    y : array-like, shape = [n_examples]
    Target values.
    Returns
    -------
    self : object
    """
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.01,size=X.shape[1] + 1)
    # self.b_ = np.float64(0.)
    self.w_[-1] = np.float64(0.) # change: absorb b into w as the last element
    # dummy = np.zeros((X.shape[0], 1))
    # dummy[0][0] = 1
    # print(dummy)
    # self.samples = np.hstack((X, dummy)) # change: absorb a 1 into X
    print(self.w_)
    # print(X)
    self.losses_ = []

    for i in range(self.n_iter): 
      net_input = self.net_input(X)
      output = self.activation(net_input)
      errors = (y - output) 
      self.w_[0:(X.shape[1])] += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
      self.w_[-1] += self.eta * 2.0 * errors.mean() # change: update b in w now
      loss = (errors**2).mean() 
      self.losses_.append(loss)
    print("after train: ", self.w_)
    return self
  
  def net_input(self, X):
    """Calculate net input"""
    # need to account for not a million bs....
    # dummy = np.zeros((X.shape[0], 1))
    # dummy[0][0] = 1
    # print(np.dot(np.hstack((X, dummy)), self.w_))
    return np.dot(X, self.w_[0:(X.shape[1])])  + self.w_[-1]# change: b is in w, remove it
  
  def activation(self, X):
    """Compute linear activation"""
    return X
  
  def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
  
"""
code from the text book for the logistic regression learning model.
below are modifications to absorb the bias into the vectors w and x.
"""
# class LogisticRegressionGD:
#   """Gradient descent-based logistic regression classifier.
#   Parameters
#   ------------
#   eta : float
#   Learning rate (between 0.0 and 1.0)
#   n_iter : int
#   Passes over the training dataset.
#   random_state : int
#   Random number generator seed for random weight
#   initialization.
#   Attributes
#   -----------
#   w_ : 1d-array
#   Weights after training.
#   b_ : Scalar
#   Bias unit after fitting.
#   losses_ : list
#   Mean squared error loss function values in each epoch.
#   """
#   def __init__(self, eta=0.01, n_iter=50, random_state=1):
#     self.eta = eta
#     self.n_iter = n_iter
#     self.random_state = random_state

#   def fit(self, X, y):
#     """ Fit training data.
#     Parameters
#     ----------
#     X : {array-like}, shape = [n_examples, n_features]
#     Training vectors, where n_examples is the
#     number of examples and n_features is the
#     number of features.
#     y : array-like, shape = [n_examples]
#     Target values.
#     Returns
#     -------
#     self : Instance of LogisticRegressionGD
#     """
#     rgen = np.random.RandomState(self.random_state)
#     self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
#     self.b_ = np.float_(0.)
#     self.losses_ = []
#     for i in range(self.n_iter):
#       net_input = self.net_input(X)
#       output = self.activation(net_input)
#       errors = (y - output)
#       self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
#       self.b_ += self.eta * 2.0 * errors.mean()
#       loss = (-y.dot(np.log(output))
#               - ((1 - y).dot(np.log(1 - output)))
#               / X.shape[0])
#       self.losses_.append(loss)
#     return self
  
#   def net_input(self, X):
#     """Calculate net input"""
#     return np.dot(X, self.w_) + self.b_
  
#   def activation(self, z):
#     """Compute logistic sigmoid activation"""
#     return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
  
#   def predict(self, X):
#     """Return class label after unit step"""
#     return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
  
# now is a given function to be able to depict the decision regions 
# of the perceptron/classifier

def plot_decision_regions(X, y, classifier, resolution=0.02):
  # setup marker generator and color map
  markers = ('o', 's', '^', 'v', '<')
  colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
  cmap = ListedColormap(colors[:len(np.unique(y))])
  # plot the decision surface
  x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
  lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) # T -> transpose
  lab = lab.reshape(xx1.shape)
  plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
  plt.xlim(xx1.min(), xx1.max())
  plt.ylim(xx2.min(), xx2.max())
  # plot class examples
  for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0],
      y=X[y == cl, 1],
      alpha=0.8,
      c=colors[idx],
      marker=markers[idx],
      label=f'Class {cl}',
      edgecolor='black')
  plt.show()
# ===================================================================
# SCRIPTING
# ===================================================================

s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'
print('From URL:', s)

df = pd.read_csv(s,
     header=None,
     encoding='utf-8')


# set up classes for setosa vs versi
y = df.iloc[0:100, 4].values # values in the 4th column of csv -> names of iris
y = np.where(y == "Iris-setosa", 0, 1) # setosa -> 0, versi 1

# extract the other information defining the classes
# specifically sepal and petal lengths
X = df.iloc[0:100, [0, 2]].values  

ada = AdalineGD(eta=0.01, n_iter=10) # note that eta needs to be small here!
ada.fit(X, y) # hand off the iris data and correct labels to learning algorithm
# plotting of the linearly separable decision regions.
plot_decision_regions(X, y, classifier=ada)