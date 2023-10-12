import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from keras.models import load_model

# "yf.pdr_override()" will allow us to use pandas_datareader syntax
# in conjunction with Yahoo's updated API

yf.pdr_override()


# Defining the date start and end point for the finance data
# Initiating API connection in accordance with updated Yahoo's documentation

start = '2010-01-01'
end = '2023-10-10'

st.title('Stock Ticker Predictor')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start, end)

# Data Description

st.subheader('Ticker Data from 2010 - 2023')
st.write(df.describe())

# Data Visualization

st.subheader('Ticker Closing Price Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'black')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Ticker Closing Price Time Chart with 100 Day Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'red')
plt.plot(df.Close, 'black')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Ticker Closing Price Time Chart with 100 and 200 Day Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'red')
plt.plot(ma200, 'green')
plt.plot(df.Close, 'black')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# This is where we split the data for TRAINING and TESTING
# 70% of data will be used for TRAINING and 30% for TESTING

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

# importing scaling tools
# Scaling data down between zero and one with MinMaxScaler
# defining object "scaler" to fit data into

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# The following block is commented out because the ML Model loaded next is trained
# Dividing data into two lists - x_train and y_train

# x_train = []
# y_train = []

# for i in range(100, data_training_array.shape[0]):
#    x_train.append(data_training_array[i-100: i])
#    y_train.append(data_training_array[i, 0])

# x_train, y_train = np.array(x_train), np.array(y_train)

# Loading LSTM Machine Learning Model "ticker_keras_model"

model = load_model('ticker_keras_model.h5')

# ML model was trained in Jupyter Notebook during creation

# Testing data with last 100 days of training data to test the model

past_100_days = data_training.tail(100)

# pandas DataFrame append method has been updated
# per new documentation - I used concat to update the final DataFrame

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

# defining x-test and y-test

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# defining prediction variable

y_predicted = model.predict(x_test)

# scaling values down

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# FINAL VISUALIZATION

st.subheader('Predictions VS Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'black', label = 'Original Stock Price')
plt.plot(y_predicted, 'blue', label = 'Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

print(f"Tomorrow's Stock Price Prediction is: {np.around(y_predicted[-1], 2)}")
