import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset= pd.read_csv('/content/drive/MyDrive/ML Dataset/train.csv')
dataset= pd.read_csv('/content/drive/MyDrive/ML Dataset/heart.csv')
dataset

dataset.shape

dataset.describe()

x_value= dataset.iloc[1:700, 3:4]
y_value= dataset.iloc[1:700, 4:5]

x_value.boxplot(column=['x'])

y_value.boxplot(column=['y'])

Scatter=plt.scatter(x_value, y_value)
plt.xlabel('BPS')
plt.ylabel('Cholesterol')
plt.show()

#cleaning dataset
clean_dataset = dataset.dropna()
clean_dataset.shape

x_value = clean_dataset['x']
y_value = clean_dataset['y']

#ML Model
def Hypothesis(theta_array, x):
  return theta_array[0] + theta_array[1]*x

#Cost Function
def costfunction(theta_array, x, y,m):
  total_cost= 0;
  for i in range(m):
    total_cost += ((theta_array[0] + theta_array[1]*x[i]) - y[i])**2
  return total_cost/(2*m)

def GradientDescent(theta_array, x, y, alpha, m):
  summation_0 = 0
  summation_1 = 0
  for i in range(m):
      summation_0 +=(theta_array[0]+(theta_array[1]*x[i])- y[i])
      summation_1 +=((theta_array[0]+(theta_array[1]*x[i])- y[i]))*x[i]

      new_theta_0 = (theta_array[0] - ((alpha/m)*summation_0))
      new_theta_1 = (theta_array[1] - ((alpha/m)*summation_1))

      improvised_theta = [new_theta_0, new_theta_1]
  print(improvised_theta)
  return improvised_theta

def training(x,y,alpha,epochs):
  theta_0=0
  theta_1=0
  m=x.size
  cost_values=[]
  theta_array=[theta_0, theta_1]
  for i in range(epochs):
      theta_array= GradientDescent(theta_array, x, y, alpha, m)
      loss=costfunction(theta_array, x, y, m)
      cost_values.append(loss)
      y_new=theta_array[0]+theta_array[1]*x
      plt.scatter(x,y,color='blue')
      plt.plot(x,y_new,color='red')
      plt.show()


  x=np.arange(0, epochs)
  plt.plot(x, cost_values)
  plt.show()

alpha=0.0001
epochs=100

x_feature = x_value.values.reshape(x_value.size)
y_feature = y_value.values.reshape(y_value.size)



training(x_feature, y_feature, alpha, epochs)