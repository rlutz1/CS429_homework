import numpy as np

"""
An oop representation of the perceptron model and algorithm.
as mentioned, the algorithm is the model, and the model is the algorithm.
fit() is the convention for the learning algorithm of the model.
predict() is used to classify new data.
"""
class Perceptron:

  """
  eta          -> the learning rate, best being small. being too big can cause 
                  instability
  n_iter       -> this many passes over the training set. why not tolerance?
  random_state -> randomizer seed such that you can get the same random
                  sequence each time.
  """
  def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

  """
  this is the actual learning algorithm.
  """
  def fit(self, X, y):
    rgen = np.random.RandomState(self.random_state) # rand num generator, LEGACY!
    # generate random weights; .normal generates from a gaussian distribution.
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.normal.html
    self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1]) 
    self.b_ = np.float_(0.) # just 0 as a float value (float_ used as to not clash with py types)
    self.errors_ = [] # empty error count tracker
    for _ in range(self.n_iter): # iterate this many times
      errors = 0
      # we are updating every single sample weight EVERY epoch
      # big difference from adaline
      # xi is the training sample, target is the correct label/class/(0 or 1)
      for xi, target in zip(X, y): # zip x with y as tuples,
        # update for the weight: w = w + eta * (predict - correct label) * xi
        update = self.eta * (target - self.predict(xi))
        self.w_ += update * xi
        # update for b: b = b + eta * (predict - correct label) 
        self.b_ += update
        errors += int(update != 0.0) # add to errors if error made, update is 0 means no misclass.
      self.errors_.append(errors) # how many misclassifications this epoch
    return self

  # use: [X (dot) w] + b
  def net_input(self, X):
    return np.dot(X, self.w_) + self.b_

  # the general prediction entry point
  def predict(self, X):
    return np.where(self.net_input(X) >= 0.0, 1, 0)