#!/usr/bin/env python
# coding: utf-8

# In[29]:


BHARAT INTERN 
SAMIKSHA CHAVAN 
K.K.WAGH COLLEGE 

# Import the required libraries
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense



# In[30]:


# Load the historical stock price data
df = pd.read_csv('C:\\Users\\samik\\Downloads\\archive (3)\\HistoricalQuotes.csv')

# Check the structure and column names of your DataFrame
print(df.head())
print(df.columns)


# In[31]:


# Identify the column containing the stock prices
# Replace 'Close' with the correct column name if necessary
column_name = None
for col in df.columns:
    if 'close' in col.lower():
        column_name = col
        break

# Prepare the data
if column_name is not None:
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\d.]', '', x))  # Remove non-numeric characters
    df[column_name] = df[column_name].astype(float)  # Convert to float

    data = df[column_name].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
else:
    raise KeyError("Column containing stock prices not found in DataFrame.")


# In[32]:


# Define the training data
train_data_len = int(len(data) * 0.8)  # 80% of the data will be used for training
train_data = scaled_data[:train_data_len]

# Create the training dataset
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the input data for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Prepare the testing data
test_data = scaled_data[train_data_len - 60:]
print (test_data)


# In[23]:


# Create the testing dataset
x_test, y_test = [], []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test, y_test)


# In[37]:


# Reshape the input data for LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_test)


# In[34]:


# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print(predictions)



# In[35]:


# Prepare the plotting data
plot_data_len = len(df) - train_data_len - 60
plot_data = df.iloc[-plot_data_len:, :]
plot_predictions = predictions[-plot_data_len:]
print(plot_predictions)


# In[36]:


# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(plot_data['Date'], plot_data[column_name], label='Actual Stock Price')
plt.plot(plot_data['Date'], plot_predictions, label='Predicted Stock Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




