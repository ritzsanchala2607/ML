# -*- coding: utf-8 -*-
"""KNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13DQjhHEBE4ZmoQ-yVmsxTNkEV5CWJZEU
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

url ="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepallength','sepalwidth','petallength','patelwidth','Class']
dataset=pd.read_csv(url,names=names)
dataset

tariningclass=dataset.iloc[:,-1]
tariningclass

uniquelist =list(set(tariningclass))
uniquelist

for i in range(len(tariningclass)):
  for j in range(len(uniquelist)):
    if(tariningclass[i]==uniquelist[j]):
      tariningclass[i]=j
tariningclass

def euclideanDistance(rowsi, rowsj):
  distance = 0.0
  for i in range(len(rowsi)):
    distance += (rowsi[i] - rowsj[i])**2
  return math.sqrt(distance)

training= dataset.values[:,0:4]

testing=dataset.values[149,0:4]

distance=[]
for i in range(len(training)):
  dist=euclideanDistance(training[i],testing)
  distance.append((dist,tariningclass[i]))

distance

distance.sort()

distance

distance[1:6]

K=5
voting=[0 for i in range(len(uniquelist))]
for i in range(1,K+1):
  voting[distance[i][1]]=voting[distance[i][1]]+1
voting

uniquelist[voting.index(max(voting))]

