# Importing the libraries
import pandas as pd  # for reading input data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler  # For Pre-processing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Read the data in file named customers.csv
df = pd.read_csv('customers.csv')
# value_counts = df['y'].value_counts()
# labels = value_counts.index
# counts = value_counts.values
# plt.pie(counts, labels=labels, autopct='%1.1f%%')
# plt.title("Distribution of y")
# plt.show()
# Pre-processing the data
input_columns = df.drop('y', axis=1)
class_column = df['y']
oversampled = RandomOverSampler(random_state=0)  # For adding random data
# Using the fit_resample to add random data
input_columns_resampled, class_column_resampled = oversampled.fit_resample(input_columns, class_column)
# Collecting the input and output data
df = pd.concat([input_columns_resampled, class_column_resampled], axis=1)
# Select the input columns
X = df.iloc[:, 0:16].values
# Select the output column
y = df.iloc[:, -1]
# Change the y value from yes/no to numbers 0 and 1 using LabelEncoder()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)  # Now y have values of 0s and 1s
# Change other column values to number using OneHotEncoder() and ColumnTransformer
# X_job = X[:, [1]]
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
#                        sparse_threshold=0)
# X_job = ct.fit_transform(X_job)
# Changing Categorical columns to numbers "Transformations"
X_cat = X[:, [1, 2, 3, 4, 6, 7, 8, 10, 15]]
# print(X_cat.shape) // (79844, 9)
orginalNumOfCols = X_cat.shape[1]
for i in range(X_cat.shape[1]):
    currNumOfCols = X_cat.shape[1]

    indexOfColumnToEncode = currNumOfCols - orginalNumOfCols + i

    ct = ColumnTransformer(transformers=
                           [('encoder',
                             OneHotEncoder(), [indexOfColumnToEncode])],
                           remainder='passthrough',
                           sparse_threshold=0)

    X_cat = ct.fit_transform(X_cat)
# Creating input matrix
X_num = X[:, [0, 5, 9, 11, 12, 13, 14]]
# Combine the numerical columns with results of Transformations using concatenate() function
X = np.concatenate((X_num, X_cat), axis=1)
# Spilt dataset into training and testing using train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Scaling dataset using StandardScaler()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Building the Neural Network
classifier = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid'),
])
# Compile the Neural Network to model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the Neural Network
Training = classifier.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
# Graph Improving of the Loss Function
plt.plot(Training.history['loss'])
plt.plot(Training.history['val_loss'])
plt.title('Improving of the Loss Function')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
# Graph Improving of the Accuracy Function
plt.plot(Training.history['accuracy'])
plt.plot(Training.history['val_accuracy'])
plt.title('Improving of the Accuracy Function')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()
# Evaluate the model
test_loss, test_accuracy = classifier.evaluate(X_test, y_test)
print("The Value of the Loss Function", test_loss)
print("The Value of the Accuracy Function", test_accuracy)
# Using the model in test data
y_pred = classifier.predict(X_test)
y_pred_binary = (y_pred > 0.5)
print(y_pred_binary)
# Evaluate the model precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", round(100 * accuracy, 2))
precision = precision_score(y_test, y_pred_binary)
print("Precision:", round(100 * precision, 2))
recall = recall_score(y_test, y_pred_binary)
print("Recall:", round(100 * recall, 2))
f1 = f1_score(y_test, y_pred_binary)
print("F1 Score:", round(100 * f1, 2))
# Graph the Confusion Matrix
confusion_mat = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(confusion_mat, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
