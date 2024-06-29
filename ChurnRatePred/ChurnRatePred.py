# Install necessary packages
!pip install --upgrade scikit-learn imbalanced-learn imblearn
!pip install basemap

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential, load_model
from keras.layers import Dense
from joblib import dump, load

# Load and display the data
df = pd.read_csv('Churn_Modelling.csv')
df.head()

# Plot the distribution of the 'Exited' column
def plot_distribution(value_counts, title):
    labels = value_counts.index
    counts = value_counts.values
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.title(title)
    plt.show()

value_counts = df['Exited'].value_counts()
plot_distribution(value_counts, 'Distribution of Exited')

# Balance the dataset using RandomOverSampler
def balance_dataset(df, target_column):
    input_columns = df.drop(target_column, axis=1)
    class_column = df[target_column]
    oversampler = RandomOverSampler(random_state=0)
    input_columns_resampled, class_column_resampled = oversampler.fit_resample(input_columns, class_column)
    return pd.concat([input_columns_resampled, class_column_resampled], axis=1)

df_balanced = balance_dataset(df, 'Exited')
class_distribution = df_balanced['Exited'].value_counts()
print(class_distribution)
plot_distribution(class_distribution, 'Distribution of Exited (Balanced)')

# Prepare features and target variables
X = df_balanced.iloc[:, 3:13].values
y = df_balanced.iloc[:, 13].values

# Encode categorical data
def encode_categorical_data(X):
    labelencoder_gender = LabelEncoder()
    X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])
    distinct_values = np.unique(X[:, 1])
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = ct.fit_transform(X)
    return X, labelencoder_gender, ct

X, labelencoder_gender, ct = encode_categorical_data(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Standardize the features
def standardize_data(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, sc

X_train, X_test, sc = standardize_data(X_train, X_test)

# Build and compile the model
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_dim = len(X_train[0])
model = build_model(input_dim)

# Train the model
history = model.fit(X_train, y_train, batch_size=10, epochs=20)

# Evaluate the model
evaluation = model.evaluate(X_test, y_test)
print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])

# Make predictions and evaluate
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5)

accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Save the model and encoders
model.save("churn_model.keras")
dump(labelencoder_gender, "churn_label_encoder.pkl")
dump(ct, "churn_column_transformer.pkl")
dump(sc, "churn_standard_scaler.pkl")

# Function to predict churn for a new customer
def predict_churn(new_customer_data, model_path, labelencoder_gender_path, column_transformer_path, scaler_path):
    labelencoder_gender_loaded = load(labelencoder_gender_path)
    ct_loaded = load(column_transformer_path)
    sc_loaded = load(scaler_path)
    model = load_model(model_path)
    
    new_customer_data[:, 2] = labelencoder_gender_loaded.transform(new_customer_data[:, 2])
    new_customer_data = ct_loaded.transform(new_customer_data)
    new_customer_data = sc_loaded.transform(new_customer_data)
    
    new_prediction_proba = model.predict(new_customer_data)
    new_prediction = (new_prediction_proba > 0.5)
    return new_prediction

# Example usage
new_customer = np.array([[600, "France", "Male", 40, 3, 60000, 2, 1, 1, 50000]])
new_prediction = predict_churn(new_customer, "churn_model.keras", "churn_label_encoder.pkl", "churn_column_transformer.pkl", "churn_standard_scaler.pkl")
print("Churn Prediction:", new_prediction)
