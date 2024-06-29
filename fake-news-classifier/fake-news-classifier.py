# Install the necessary packages
!pip install nltk keras tensorflow

# Import required libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('stopwords')

# Load and display the data
df = pd.read_csv('train.csv')
df.fillna('unavailable', inplace=True)

# Combine columns to create a single text feature
df['comb'] = df['author'] + " " + df['title'] + " " + df['text']

# Display data information
df.info()

# Text preprocessing functions
stemmer = PorterStemmer()

def clean(text):
    """
    Clean the input text by removing non-alphabetic characters,
    converting to lowercase, removing stopwords, and applying stemming.
    """
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in set(stopwords.words("english"))]
    return ' '.join(text)

# Apply text cleaning
df['comb'] = df['comb'].apply(clean)
df.head()

# One-hot encode the text data
voc_size = 50000
text = df['comb']
one_hot_result = [one_hot(words, voc_size) for words in text]

# Pad the sequences
max_len = 500
X = pad_sequences(one_hot_result, padding='post', maxlen=max_len)

# Extract the labels
y = df['label'].values

# Split the data into training and validation sets
X_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

# Build the LSTM model
def build_lstm_model(voc_size, max_len):
    """
    Build and compile a Bidirectional LSTM model for text classification.
    """
    model = Sequential()
    model.add(Input(shape=max_len))
    model.add(Embedding(input_dim=voc_size, output_dim=50, input_length=max_len))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_lstm_model(voc_size, max_len)

# Train the model
history = model.fit(X_train, y_train, validation_data=(x_valid, y_valid), epochs=20, batch_size=124)

# Plot training and validation accuracy
def plot_training_history(history):
    """
    Plot the training and validation accuracy over epochs.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['accuracy'], label='train accuracy')
    ax.plot(history.history['val_accuracy'], label='val accuracy')
    ax.set_title('Model Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='upper left')
    plt.show()

plot_training_history(history)

# Save the model
model.save('fake_news_model.keras')

# Function to classify a single news article
def classify_news(author, title, text, model_path, voc_size, max_len):
    """
    Classify a news article as fake or not fake using the trained model.
    """
    model = load_model(model_path)
    news = author + " " + title + " " + text
    news_clean = clean(news)
    news_onehot = one_hot(news_clean, voc_size)
    news_seq = pad_sequences([news_onehot], padding='post', maxlen=max_len)
    prediction = model.predict(news_seq)
    return "Fake news!" if prediction[0] > 0.5 else "Not fake news!"

# Example usage
title = "House Dem Aide: We Didn’t Even See Comey’s Letter Until Jason Chaffetz Tweeted It"
author = "Darrell Lucus"
text = '''
House Dem Aide: We Didn’t Even See Comey’s Letter Until Jason Chaffetz Tweeted It By Darrell Lucus on October 30, 2016 
Subscribe Jason Chaffetz on the stump in American Fork, Utah ( image courtesy Michael Jolley, available under a Creative Commons-BY license) 
With apologies to Keith Olbermann, there is no doubt who the Worst Person in The World is this week–FBI Director James Comey. But according to a House Democratic aide, it looks like we also know who the second-worst person is as well. It turns out that when Comey sent his now-infamous letter announcing that the FBI was looking into emails that may be related to Hillary Clinton’s email server, the ranking Democrats on the relevant committees didn’t hear about it from Comey. They found out via a tweet from one of the Republican committee chairmen. 
As we now know, Comey notified the Republican chairmen and Democratic ranking members of the House Intelligence, Judiciary, and Oversight committees that his agency was reviewing emails it had recently discovered in order to see if they contained classified information. Not long after this letter went out, Oversight Committee Chairman Jason Chaffetz set the political world ablaze with this tweet. FBI Dir just informed me, "The FBI has learned of the existence of emails that appear to be pertinent to the investigation."
'''

print(classify_news(author, title, text, 'fake_news_model.keras', voc_size, max_len))
