# -*- coding: utf-8 -*-
"""Random_Forest.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mkUu7tuU-1dPFQpGG8wC65c3QF-VdLea
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

iris=datasets.load_iris()

iris

class_name=iris.target_names

class_name

features=iris.feature_names

features

data=pd.DataFrame({
                  'sepal length':iris.data[:,0],
                  'sepal width':iris.data[:,1],
                  'petal length':iris.data[:,2],
                  'width length':iris.data[:,3],
                  'species':iris.target
})
data

from sklearn.model_selection import train_test_split

x=data[['sepal length', 'sepal width', 'petal length', 'width length']]
y=data[['species']]

X_train,X_Test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=100)
forest.fit(X_train,y_train)

y_pred=forest.predict(X_Test)

from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)

import pydot
from sklearn.tree import export_graphviz

tree=forest.estimators_[5]

export_graphviz(tree,out_file='/content/tree5.dot',feature_names=features)

(graph,)=pydot.graph_from_dot_file('/content/tree5.dot')

graph.write_png('/content/tree5.png')

forest.feature_importances_

pd.Series(forest.feature_importances_,index=features)
