"""
Compare the performance of Adaline and logistic regression 
(bias absorbed versions) on the Iris
and Wine datasets that can be obtained from the UCI machine 
learning repository. You may use the Python
program given in our textbook (Page 117) to import the datasets.

• Iris dataset - You may consider the samples with the labels setosa, 
versicolor to form a training set for binary classification.

• Wine dataset - You may consider the samples with in the first 
two classes (1 and 2) to form a training set for binary classification.

The comparisons should be done based on the convergence of the loss. 

In order to make apple-to-apple comparisons, 
you should use the same hyperparameters 
and number of epochs for both learning algorithms.
"""

from log_ada_absorbed_bias import AdalineGD, LogisticRegressionGD
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
