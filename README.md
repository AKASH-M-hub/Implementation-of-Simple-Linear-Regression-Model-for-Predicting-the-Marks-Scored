# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Akash M
RegisterNumber:  212224230013
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from google.colab import files
uploaded = files.upload()

import pandas as pd
import io

df = pd.read_csv(io.BytesIO(uploaded['students_scores.csv']))

print(df)                               # Print the full DataFrame

df.head(0)                              # Show header only
df.tail(0)                              # Show header only

print(df.head())                        # Display first 5 rows
print(df.tail())                        # Display last 5 rows
x = df.iloc[:, :-1].values              # Extracting Hours (features)
print(x)

y = df.iloc[:, 1].values                # Extracting Scores (target)
print(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0
)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

print("Predicted values:", y_pred)
print("Actual values:", y_test)
plt.scatter(x_train, y_train, color='black')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test, y_test, color='black')
plt.plot(x_train, regressor.predict(x_train), color='red')  # line stays the same
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse = mean_absolute_error(y_test, y_pred)
print('MSE =', mse)

mae = mean_absolute_error(y_test, y_pred)
print('MAE =', mae)

rmse = np.sqrt(mse)
print("RMSE =", rmse)

```
## Output:
![simple linear regression model for predicting the marks scored](sam.png)

![image](https://github.com/user-attachments/assets/d636f512-59af-475c-8ee2-bc9d4a20e042)

![image](https://github.com/user-attachments/assets/fcdf20c7-fd74-4041-9faf-3c134ef7181b)

![image](https://github.com/user-attachments/assets/91a74ac8-4a36-4704-9dde-461eda049f39)

![image](https://github.com/user-attachments/assets/5d2bcf53-8b83-41be-bb1b-8e5edb403c83)

![image](https://github.com/user-attachments/assets/436333d7-bccb-40a6-9cdf-cfc69531dd22)

![image](https://github.com/user-attachments/assets/2b183bea-cc77-4cf7-8f97-6bdb683f484b)

![image](https://github.com/user-attachments/assets/1e397e19-94d6-4a11-901d-c265aab945fb)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
