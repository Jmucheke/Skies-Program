# Step 1.0 Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Step 2.0 Providing data
x = np.array([5,15,25,35,45,55]).reshape((-1,1))
#y = np.array([8.33,13.733,19.1333,24.5333,29.9333,35.33333])
y = np.array([5,20,14,32,22,38])

print(f"===============These are the values in x:==============\n {x}")
print(f"===============These are the values in y:==============\n {y}")

# Step 3.0 create the Model
Ex4_1=LinearRegression()

# Step 4.0 Fit the data in the model
Ex4_1.fit(x,y)

# Step 5.0 Obtain the results
r_sq = Ex4_1.score(x,y)
print(r_sq)

# Step 6.0 Interpret the result

b0 = Ex4_1.intercept_ # y intercept
b1 = Ex4_1.coef_  #coefficient

print(f"=============== The y-intercept is:===========\n {b0}")
print(f"===============The gradient is:==============\n {b1}")

# Step 7.0 Apply the results

# Step 7.1 Using Existing Values

y_pred = Ex4_1.predict(x)
print(f"========== The y prediction is:==========\n {y_pred}")

# Step 7.2 Forecast values
test_data = np.arange(6).reshape((-1,1))
print(f"================Test data is:=============\n {test_data}")

y_new = Ex4_1.predict(test_data)
print(f"==================The new predicted values are:=============== \n {y_new}")
