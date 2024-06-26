# Importing the necessary libraries
import json
import random
import numpy as np
import nltk
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pickle
from textblob import TextBlob

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('omw-1.4')

# Load intents JSON file
with open('Q:\Ai Courses\AI Hsoub\DeepLearning\Course Project\Jp_notebook\(15) بناء بوت محادثة\X-03\intents.json') as file:
    data = json.load(file)

# Initialize variables
vocab = []
classes = []
doc_X = []
doc_y = []

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Tokenizer function
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Process intents
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each pattern
        tokens = tokenize(pattern)
        vocab.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["class"])
    if intent["class"] not in classes:
        classes.append(intent["class"])

# Remove punctuation and stopwords, and lemmatize
vocab = [word for word in vocab if word not in string.punctuation]
vocab = [lemmatizer.lemmatize(word.lower()) for word in vocab]
stop_words = set(stopwords.words('english'))
vocab = [word for word in vocab if word not in stop_words]

# Remove duplicates and sort
vocab = sorted(set(vocab))
classes = sorted(set(classes))

# Create training data
X = []
y = []

for idx, doc in enumerate(doc_X):
    bow = []
    doc = lemmatizer.lemmatize(doc.lower())
    for word in vocab:
        bow.append(1) if word in doc else bow.append(0)
    X.append(bow)
    output_row = [0] * len(classes)
    cl = doc_y[idx]
    cl_index = classes.index(cl)
    output_row[cl_index] = 1
    y.append(output_row)

# Convert lists to arrays for the neural network
X = np.array(X)
y = np.array(y)

# Emotion detection function
def detect_emotion(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity == 0:
        return "neutral"
    else:
        return "negative"

# Add emotion to the training data
emotions = ["positive", "neutral", "negative"]
for idx, doc in enumerate(doc_X):
    emotion = detect_emotion(doc)
    doc_X[idx] += f" {emotion}"

# Recreate training data with emotion
X = []
y = []

for idx, doc in enumerate(doc_X):
    bow = []
    doc = lemmatizer.lemmatize(doc.lower())
    for word in vocab + emotions:
        bow.append(1) if word in doc else bow.append(0)
    X.append(bow)
    output_row = [0] * len(classes)
    cl = doc_y[idx]
    cl_index = classes.index(cl)
    output_row[cl_index] = 1
    y.append(output_row)

# Convert lists to arrays for the neural network
X = np.array(X)
y = np.array(y)

# Building the Neural Network
input_shape = (len(X[0]),)
output_shape = len(y[0])

model = Sequential([
    Input(shape=input_shape),
    Dense(1024, activation='relu'),
    Dropout(0.7),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(output_shape, activation='softmax'),
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model_with_emotion.h5')
