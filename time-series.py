import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Generate random time series data

np.random.seed(42)
dates = pd.date_range(start='2022-01-01', periods=100,freq='D')

values = np.random.randn(100).cumsum()

# Create a Dataframe from the generated data
data = pd.DataFrame({'date':dates,'value':values})

# Set the 'date' column as the index
data.set_index('date',inplace=True)

#Plot the time series data
plt.plot(data.index, data['value'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.title('Time Series Data')
plt.show()

# Testing for Stationarity

from statsmodels.tsa.stattools import adfuller

# Assuming 'data' is the time series data

result = adfuller(data)
print(f"===========ADF Statistics:============\n {result[0]}")

print(f"===========p-value:==============\n {result[1]}")
