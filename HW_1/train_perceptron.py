from perceptron import Perceptron
import os
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
# print(X)

# we're gonna plot these badboys.
# NO perceptron learning at this state.

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# let's train the perceptron with this data

