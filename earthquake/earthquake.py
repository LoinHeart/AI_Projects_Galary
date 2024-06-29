# Install the necessary packages
!pip install basemap

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from mpl_toolkits.basemap import Basemap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.models import load_model
import pickle

# Load and display the data
df = pd.read_csv("dataset.csv")
df = df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
df.head()

# Function to convert date and time to timestamps
def convert_to_timestamp(df):
    timestamps = []
    for _, row in df.iterrows():
        try:
            full_date_string = row['Date'] + " " + row['Time']
            full_date = datetime.datetime.strptime(full_date_string, '%m/%d/%Y %H:%M:%S')
            timestamp = full_date.timestamp()
            timestamps.append(timestamp)
        except Exception:
            timestamps.append('ValueError')
    return timestamps

# Convert date and time to timestamp
df['Timestamp'] = convert_to_timestamp(df)
df = df.drop(['Date', 'Time'], axis=1)
df = df[df.Timestamp != 'ValueError']
df['Timestamp'] = df['Timestamp'].astype(float)
df.head()

# Plot the affected areas on a map
def plot_affected_areas(df):
    m = Basemap()
    longitudes = df["Longitude"].tolist()
    latitudes = df["Latitude"].tolist()
    x, y = m(longitudes, latitudes)

    fig = plt.figure(figsize=(12, 10))
    plt.title("All affected areas")
    m.plot(x, y, "o", markersize=2, color='blue')
    m.drawcoastlines()
    m.fillcontinents(color='coral', lake_color='aqua')
    m.drawmapboundary()
    m.drawcountries()
    plt.show()

plot_affected_areas(df)

# Prepare features and target variables
X = df[['Latitude', 'Longitude', 'Timestamp']].values
y = df[['Depth', 'Magnitude']].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

model = build_model(X_train.shape[1])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Plot training and validation loss
def plot_training_history(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss by Epoch')
    plt.legend()
    plt.show()

plot_training_history(history)

# Save the model and scaler
model.save('earthquake_model.keras')

scaler_path = 'earthquake_scaler.pkl'
with open(scaler_path, 'wb') as file:
    pickle.dump(scaler, file)

# Function to predict depth and magnitude
def predict_earthquake(latitude, longitude, model_path, scaler_path):
    model = load_model(model_path)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    
    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.timestamp()
    new_data = np.array([[latitude, longitude, timestamp]])
    scaled_data = scaler.transform(new_data)
    predictions = model.predict(scaled_data)
    Depth = predictions[0][0]
    Magnitude = predictions[0][1]
    return Depth, Magnitude

# Example usage
latitude = 19
longitude = 145
depth, magnitude = predict_earthquake(latitude, longitude, 'earthquake_model.keras', 'earthquake_scaler.pkl')
print("Predicted Depth: ", depth)
print("Predicted Magnitude: ", magnitude)
