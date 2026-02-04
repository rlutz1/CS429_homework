from adaline import AdalineGD
import os
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'
print('From URL:', s)

df = pd.read_csv(s,
     header=None,
     encoding='utf-8')
print(df.head()) # just prints out the last 5 entries of the csv as confirmation.

"""
the tasks following.
(1) we're going to extract the first 100 classs labels 
that correspond to 50 iris-setosa, 50 iris-versicolor
(2) 1 -> versicolor
    0 -> setosa
(3) those 1/0's are assigned to a vector y; 
(4) X is then set to the other values defining that class
"""

# (1) & (2) -> set up classes for setosa vs versi
y = df.iloc[0:100, 4].values # values in the 4th column of csv -> names of iris
y = np.where(y == "Iris-setosa", 0, 1) # (3) setosa -> 0, other 1
# print(y)

#(4) extract the other information defining the classes
# specifically sepal and petal lengths
X = df.iloc[0:100, [0, 2]].values  
# print(X.shape)

# TRAIN THE MODEL
ada = AdalineGD(eta=0.001, n_iter=100)
ada.fit(X, y) # hand off the iris data and correct labels to learning algorithm


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
  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
  np.arange(x2_min, x2_max, resolution))
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
    
# plotting of the linearly separable decision regions.
plot_decision_regions(X, y, classifier=ada)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()
