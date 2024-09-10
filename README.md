# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Load necessary libraries for data handling, metrics, and visualization.
2.Load Data: Read the dataset using pd.read_csv() and display basic information.

3.Initialize Parameters: Set initial values for slope (m), intercept (c), learning rate, and epochs.

4.Gradient Descent: Perform iterations to update m and c using gradient descent.

5.Plot Error: Visualize the error over iterations to monitor convergence of the model.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SURYAMALARV
RegisterNumber: 212223230224
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```


## Output:
![image](https://github.com/user-attachments/assets/7ef6a504-ec83-4ec4-83ff-4471cfe2e4be)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/32efa0c6-acca-4b1a-ba41-3e5c5ff37b83)

```
#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)
```
![image](https://github.com/user-attachments/assets/a813778a-5d35-48cc-9172-5a29920bfac9)
```
m=0
c=0
L=0.001
epochs=5000
n=float(len(X))
error=[]
for i in range(epochs):
  Y_pred=m*X+c
  D_m=(-2/n)*sum(X*(Y-Y_pred))
  D_c=(-2/n)*sum(Y-Y_pred)
  m=m-L*D_m
  c=c-L*D_c
  error.append(sum(Y-Y_pred)**2)
print(m,c)
type(error)
print(len(error))
#print(error)
plt.plot(range(0,epochs),error)
```
![image](https://github.com/user-attachments/assets/ff306b0b-c4d7-4f08-a119-5ce849d77c67)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
