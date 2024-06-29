# Install the necessary package
"""!pip install yfinance"""

# Import required libraries
from datetime import date, timedelta
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM

# Set the number of days for historical data
days = 100

# Calculate the start and end dates
today = date.today()
end_date = today.strftime("%Y-%m-%d")
start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")

# Fetch Bitcoin historical data from Yahoo Finance
data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)

# Prepare the data
data["Date"] = data.index
data.reset_index(drop=True, inplace=True)

# Plot the historical prices
def plot_historical_prices(data):
    dates = data['Date']
    prices = data['Adj Close']

    plt.figure(figsize=(10, 6))
    plt.plot(dates, prices, color='green', label='Known Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Bitcoin Prices')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

plot_historical_prices(data)

# Function to create a dataset for training the model
def create_dataset(serie, window_size=20):
    dataX, dataY = [], []
    for i in range(len(serie) - window_size - 1):
        a = serie[i:(i + window_size), 0]
        dataX.append(a)
        dataY.append(serie[i + window_size, 0])
    return np.array(dataX), np.array(dataY)

# Preprocess the data
window_size = 20
closedf = data[['Adj Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
closedf = scaler.fit_transform(closedf)
X, y = create_dataset(closedf, window_size)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Build and compile the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=32, dropout=0.1, activation="relu"))
    model.add(Dense(1, activation="relu"))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse'])
    return model

model = build_model((window_size, 1))

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=8)

# Plot the training and validation MSE
def plot_training_history(history):
    train_mse = history.history['mse']
    val_mse = history.history['val_mse']
    epochs = range(1, len(train_mse) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mse, 'b', label='Training MSE')
    plt.plot(epochs, val_mse, 'r', label='Validation MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

plot_training_history(history)

# Function to make future price predictions
def predict_future_prices(model, X, scaler, pred_steps=10):
    predicted_prices = []
    X_pred = [X[-1]]
    X_pred = np.array(X_pred)

    for _ in range(pred_steps):
        prediction = model.predict(X_pred)
        predicted_prices.append(prediction[0])
        X_pred = np.append(X_pred, prediction, axis=1)
        X_pred = X_pred[:, 1:]

    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

predicted_prices = predict_future_prices(model, X, scaler)

# Plot the known and predicted prices
def plot_prediction(data, predicted_prices, previous_days=20, pred_steps=10):
    dates_known = data["Date"].iloc[-previous_days:].values
    known_prices = data["Adj Close"].iloc[-previous_days:].values
    dates_pred = pd.date_range(start=today + timedelta(days=1), periods=pred_steps).values

    plt.figure(figsize=(10, 6))
    plt.plot(dates_known, known_prices, color='green', label='Known Prices')
    plt.plot(dates_pred, predicted_prices, color='red', label='Predicted Prices')
    plt.plot([dates_known[-1], dates_pred[0]], [known_prices[-1], predicted_prices[0][0]], color='blue')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Bitcoin Price Prediction')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

plot_prediction(data, predicted_prices)

# Summary of the model
model.summary()
