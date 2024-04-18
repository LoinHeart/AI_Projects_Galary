import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
from joblib import dump, load
# Read the Data from csv file
data = pd.read_csv('housepricedata.csv')
X = data.values[:, 0:10]
Y = data.values[:, 10]
# Creating Scaler variable named min_max_scaler to change value between 0,1
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
# split the data train, validation, test
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.2)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
# Building the Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(10,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid'),
])
# Compile the Neural Network to model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# View the model
model.summary()
# Train the Neural Network
Training = model.fit(X_train, Y_train, validation_data=(X_val_and_test, Y_val_and_test), epochs=100, batch_size=8)
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
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print("The Value of the Loss Function", test_loss)
print("The Value of the Accuracy Function", test_accuracy)
# Evaluate the model precision_score, recall_score, f1_score
y_pred = model.predict(X_test)
y_pred_classes=[]
for prob in y_pred:
    if prob >= 0.5:
        y_pred_classes.append(1)
    else:
        y_pred_classes.append(0)
precision = precision_score(Y_test, y_pred_classes)
recall = recall_score(Y_test, y_pred_classes)
f1 = f1_score(Y_test, y_pred_classes)
print("Precision:", round(precision*100,0))
print("Recall:", round(recall*100,0))
print("F1 Score:", round(f1*100,0))
# Graph the Confusion Matrix
confusion_mat = confusion_matrix(Y_test, y_pred_classes)
sns.heatmap(confusion_mat, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# Save the Trained Neural Network as Model
model.save("houses_model.keras")
# Save The Scaler variable
dump(min_max_scaler, "houses_min_max_scaler.pkl")





















