# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
##DEVELOPED BY:NAINA MOHAMED Z
##REG NO:212223230131
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
### Dataset
![image](https://github.com/user-attachments/assets/efd84165-c37b-40c8-ac5b-b0960ba4e534)

### Head Values
![image](https://github.com/user-attachments/assets/34826b8e-c6cf-40d7-9a1c-8b219af1c123)

### Tail Values
![image](https://github.com/user-attachments/assets/940fdce2-3026-41f9-9bfd-9f1eec12daec)

### X and Y values
![image](https://github.com/user-attachments/assets/91bd49d2-0828-4c63-bba8-d454c41f0f1b)

### Predication values of X and Y
![image](https://github.com/user-attachments/assets/dea9b31c-5250-4caa-b7f6-56c723e21f63)

### MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/55700094-6484-4952-95fc-cc27afac7f67)

### Training Set
![image](https://github.com/user-attachments/assets/0543341b-437e-4554-a194-220550518607)

### Testing Set
![image](https://github.com/user-attachments/assets/357ffc49-405f-4558-afed-7446b07e7da8)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
