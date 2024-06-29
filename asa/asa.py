import pandas as pd

# Load the datasets
tweets = pd.read_csv('tweets.csv', encoding = "utf-8")
positive = pd.read_csv('positive.csv' ,encoding = "utf-8")
negative = pd.read_csv('negative.csv' ,encoding = "utf-8")

# Display the first few rows of the datasets
print(tweets.head())
print(positive.head())
print(negative.tail())

# Import necessary libraries for text processing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from snowballstemmer import stemmer
import string

# Function to remove specified characters from text
def remove_chars(text, del_chars):
    for char in del_chars:
        text = text.replace(char, "")
    return text

# Function to clean text by removing punctuations, numbers, and extra spaces
def cleaningText(text):
    numbers="0123456789"
    arabic_punctuation='''`÷×؛<>_()*^ـ،/:"؟.,'~¦+|!”…“–ـ'''
    english_punctuation=string.punctuation
    
    del_chars=english_punctuation+arabic_punctuation+numbers
    text = remove_chars(text, del_chars)
     
    text = text.replace('\n', ' ')  
    text = text.strip(' ')  
    return text

# Function to filter out stopwords from tokenized text
def filteringText(tokens_list):  
    listStopwords = stopwords.words('arabic')
    filtered = [txt for txt in tokens_list if txt not in listStopwords]
    return filtered

# Function to convert a list of words back into a sentence
def toSentence(words_list):  
    sentence = ' '.join(word for word in words_list)
    return sentence

# Function to stem words in a tokenized text
def stemmingText(tokens_list): 
    ar_stemmer = stemmer("arabic")
    tokens_list_stem = [ar_stemmer.stemWord(word) for word in tokens_list]
    return tokens_list_stem

# Demonstration of stemming
ar_stemmer = stemmer("arabic")
print(ar_stemmer.stemWord("رايع"))
print(ar_stemmer.stemWord("رائع"))
print(ar_stemmer.stemWord("رائعون"))
print(ar_stemmer.stemWord("رائعين"))

# Example of text cleaning and processing
text = "!أنا أحب الذهاب إلى الحديقة، كل يوم 9 صباحاً مع رفاقي هؤلاء "
print(text)
text = cleaningText(text)
print(text)
tokens_list = word_tokenize(text)
print(tokens_list)
tokens_list = filteringText(tokens_list)
print(tokens_list)
tokens_list = stemmingText(tokens_list)
print(tokens_list)

# Function to preprocess text
def text_preprocessing(text):
    text = cleaningText(text)
    tokens = word_tokenize(text)
    tokens = filteringText(tokens)
    tokens = stemmingText(tokens)
    return tokens

# Apply preprocessing to the tweets
tweets['tweet_preprocessed'] = tweets['tweet'].apply(text_preprocessing)

# Remove duplicates from preprocessed tweets
tweets.drop_duplicates(subset = 'tweet_preprocessed', inplace = True)

# Apply preprocessing to positive and negative words
positive['word_preprocessed'] = positive['word'].apply(text_preprocessing).apply(toSentence)
positive.drop_duplicates(subset = 'word_preprocessed', inplace = True)
positive.dropna(subset = 'word_preprocessed', inplace = True)

negative['word_preprocessed'] = negative['word'].apply(text_preprocessing).apply(toSentence)
negative.drop_duplicates(subset = 'word_preprocessed', inplace = True)
negative.dropna(subset = 'word_preprocessed', inplace = True)

# Create dictionaries for positive and negative words with their scores
dict_positive = {row['word_preprocessed'].strip(): int(row['score']) for _, row in positive.iterrows()}
dict_negative = {row['word_preprocessed'].strip(): int(row['score']) for _, row in negative.iterrows()}

# Function to calculate the polarity score of a list of words
def get_polarity(words_list):
    score = sum(dict_positive.get(word, 0) for word in words_list) + sum(dict_negative.get(word, 0) for word in words_list)
    if score > 0:
        polarity = 'positive'
    elif score < 0:
        polarity = 'negative'
    else:
        polarity = 'neutral'
    return score, polarity

# Calculate polarity scores for tweets
for idx, row in tweets.iterrows():
    tweets_words = tweets.loc[idx, 'tweet_preprocessed']
    score, polarity = get_polarity(tweets_words)
    tweets.loc[idx, 'polarity_score'] = score
    tweets.loc[idx, 'polarity'] = polarity

# Plot the distribution of tweet polarities
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (6, 6))
x = tweets['polarity'].value_counts()
labels = tweets['polarity'].value_counts().index
explode = (0.1, 0, 0)
ax.pie(x = x, labels = labels, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
ax.set_title('Tweets Polarities', fontsize = 16, pad = 20)
plt.show()

# Show the top positive and negative tweets
positive_tweets = tweets[tweets['polarity'] == 'positive'].sort_values(by = 'polarity_score', ascending=False)
print(positive_tweets[['tweet','polarity_score']].head())
negative_tweets = tweets[tweets['polarity'] == 'negative'].sort_values(by = 'polarity_score', ascending=True)
print(negative_tweets[['tweet','polarity_score']].head())

# Create word clouds for tweets and positive/negative words
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display

def create_wordcloud(text, font_path, width, height, background_color, colormap=None):
    reshaped_text = arabic_reshaper.reshape(text)
    artext = get_display(reshaped_text)
    return WordCloud(font_path=font_path, width=width, height=height, background_color=background_color, colormap=colormap, min_font_size=10).generate(artext)

# Word cloud for tweets
list_words = ' '.join([' '.join(tweet) for tweet in tweets['tweet_preprocessed'][:100]])
wordcloud = create_wordcloud(list_words, 'DroidSansMono.ttf', 600, 400, 'black')
fig, ax = plt.subplots(figsize = (8, 6))
ax.set_title('Word Cloud of Tweets', fontsize = 18)
ax.grid(False)
ax.imshow(wordcloud)
ax.axis('off')
plt.show()

# Word clouds for positive and negative words
fig, ax = plt.subplots(1, 2, figsize = (12, 10))
positive_words = ' '.join(positive['word'].values)
negative_words = ' '.join(negative['word'].values)
wordcloud_positive = create_wordcloud(positive_words, 'DroidSansMono.ttf', 800, 600, 'black', 'Greens')
wordcloud_negative = create_wordcloud(negative_words, 'DroidSansMono.ttf', 800, 600, 'black', 'Reds')

ax[0].set_title('Positive Words', fontsize = 14)
ax[0].grid(False)
ax[0].imshow(wordcloud_positive)
ax[0].axis('off')

ax[1].set_title('Negative Words', fontsize = 14)
ax[1].grid(False)
ax[1].imshow(wordcloud_negative)
ax[1].axis('off')

plt.show()

# Prepare text data for training a machine learning model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize the text data
sentences = tweets['tweet_preprocessed'].apply(toSentence)
max_words = 5000
sequence_max_len = 50
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(sentences.values)
seq = tokenizer.texts_to_sequences(sentences.values)
X = pad_sequences(seq, maxlen=sequence_max_len)

# Encode the polarities
polarity_encode = {'negative': 0, 'neutral': 1, 'positive': 2}
y = tweets['polarity'].map(polarity_encode).values

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Calculate class weights to handle class imbalance
import numpy as np
posCount = np.count_nonzero(y_train == 2)
neuCount = np.count_nonzero(y_train == 1)
negCount = np.count_nonzero(y_train == 0)
total_3 = (posCount + negCount + neuCount) / 3
weight_for_0 = total_3 / negCount
weight_for_1 = total_3 / neuCount
weight_for_2 = total_3 / posCount
class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}

# Build and train a neural network model
from keras.models import Sequential
from keras.layers import Input, Embedding, Dense, LSTM
from keras.optimizers import Adam

def create_model(embed_dim=32, hidden_unit=16, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embed_dim, input_length=sequence_max_len))
    model.add(LSTM(units=hidden_unit, dropout=dropout_rate))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

# Initial training of the model
model = create_model(embed_dim=32, hidden_unit=16, dropout_rate=0.2, learning_rate=0.001)
epochs = 10
batch_size = 128
history = model.fit(X_train, y_train, class_weight=class_weight, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Print the final training and validation accuracy
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
print("Train Accuracy:", train_accuracy[-1])
print("Validation Accuracy:", val_accuracy[-1])

# Plot model accuracy
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['accuracy'], label='train accuracy')
ax.plot(history.history['val_accuracy'], label='val accuracy')
ax.set_title('Model Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend(loc='upper left')
plt.show()

# Save the model and tokenizer for future use
model.save('sa_model.keras')
import pickle
tokenizer_path = 'sa_tokenizer'
with open(tokenizer_path, 'wb') as file:
    pickle.dump(tokenizer, file)

# Function to classify new tweets using the saved model and tokenizer
from keras.models import load_model
import pickle

def classify_tweets(tweets, model_path, tokenizer_path, sequence_max_len):
    processed_tweets = [toSentence(text_preprocessing(x)) for x in tweets]
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    text_2_seq = tokenizer.texts_to_sequences(processed_tweets)
    text_pad = pad_sequences(text_2_seq, maxlen=sequence_max_len)
    predictions = model.predict(text_pad)
    sentiments = ['Negative' if np.argmax(pred) == 0 else 'Neutral' if np.argmax(pred) == 1 else 'Positive' for pred in predictions]
    return sentiments

# Test the classify_tweets function
new_tweets = ["مكان وسخ", "مكان رائع وجميل", "من أجمل الأماكن التي زرتها بحياتي", "يمكن التسوق فيها"]
predictions = classify_tweets(new_tweets, 'sa_model.keras', 'sa_tokenizer', 50)
for sentence, prediction in zip(new_tweets, predictions):
    print(f"Sentence: {sentence} - Sentiment: {prediction}")
