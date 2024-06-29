# Install the necessary packages
!pip install yfinance nltk seaborn keras tensorflow snowballstemmer imblearn

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from snowballstemmer import stemmer
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load and display the data
tweets = pd.read_csv('tweets.csv', encoding="utf-8")
print('Data size:', tweets.shape)
tweets.head()

# Plot the topic distribution
def plot_topic_distribution(data):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, y='topic')
    plt.title('Topics Distribution', fontsize=18)
    plt.show()

plot_topic_distribution(tweets)

# Clean the tweets
def remove_chars(text, del_chars):
    for char in del_chars:
        text = text.replace(char, "")
    return text

def remove_repeating_char(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

def clean_tweet(tweet):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  
                      u"\U0001F300-\U0001F5FF"  
                      u"\U0001F680-\U0001F6FF"  
                      u"\U0001F1E0-\U0001F1FF" 
                      u"\U00002500-\U00002BEF"  
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642" 
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  
                      u"\u3030"
                      u"\u2066"
                      "]+", re.UNICODE)
    
    tweet = re.sub(emoj, '', tweet)
    tweet = re.sub("@[^\s]+", "", tweet)
    tweet = re.sub("RT", "", tweet)
    tweet = re.sub(r"(?:\|http?\://|https?\://|www)\S+", "", tweet)
    tweet = re.sub(r'[0-9]+', '', tweet)
    tweet = remove_chars(tweet, "٠١٢٣٤٥٦٧٨٩")
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    tweet = remove_chars(tweet, punctuations_list)
    tweet = remove_repeating_char(tweet)
    tweet = tweet.replace('\n', ' ')
    tweet = tweet.strip(' ')
    return tweet

# Tokenization, filtering, and stemming functions
def tokenizing_text(text):
    return word_tokenize(text)

def filtering_text(tokens_list):
    list_stopwords = stopwords.words('arabic')
    return [txt for txt in tokens_list if txt not in list_stopwords]

def stemming_text(tokens_list):
    ar_stemmer = stemmer("arabic")
    return [ar_stemmer.stemWord(word) for word in tokens_list]

def to_sentence(words_list):
    return ' '.join(word for word in words_list)

# Full tweet processing pipeline
def process_tweet(tweet):
    tweet = clean_tweet(tweet)
    tokens_list = tokenizing_text(tweet)
    tokens_list = filtering_text(tokens_list)
    tokens_list = stemming_text(tokens_list)
    return tokens_list

# Process a sample tweet
text = "أنا أحب الذهاب إلى الحديقة 🌝، كل يوم 9 صباحاً، مع رفاقي هؤلاء! @toto  "
processed_tweet = process_tweet(text)
print(processed_tweet)

# Apply tweet processing to the dataset
tweets['tweet'] = tweets['tweet'].apply(process_tweet)

# Oversample to balance the classes
def oversample_data(tweets):
    oversample = RandomOverSampler()
    return oversample.fit_resample(tweets, tweets.topic)

tweets, Y = oversample_data(tweets)
plot_topic_distribution(tweets)

# Encode the target variable
def encode_labels(tweets):
    le_topics = LabelEncoder()
    tweets['topic'] = tweets[['topic']].apply(le_topics.fit_transform)
    classes = le_topics.classes_
    return tweets, le_topics, classes

tweets, le_topics, classes = encode_labels(tweets)
print("No. of classes:", len(classes))
print("Classes:", classes)
print("Coding: ", le_topics.transform(classes))

# Convert token lists to sentences
sentences = tweets['tweet'].apply(to_sentence)

# Tokenize the sentences
def tokenize_sentences(sentences, max_words=10000):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(sentences)
    return tokenizer, tokenizer.texts_to_sequences(sentences)

tokenizer, X = tokenize_sentences(sentences)
print("Number of words:", len(tokenizer.word_counts))

# Pad sequences to ensure uniform length
def pad_sequences_to_max_len(X, max_len=50):
    return pad_sequences(X, maxlen=max_len)

X = pad_sequences_to_max_len(X)
print(X[0])
print(X[1])

# Split the data into training and testing sets
y = tweets['topic']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Build the LSTM model
def build_lstm_model(max_words, embed_dim=32, hidden_unit=16, dropout_rate=0.2, max_len=50, n_classes=7):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embed_dim, input_length=max_len))
    model.add(LSTM(units=hidden_unit, dropout=dropout_rate))
    model.add(Dense(units=n_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_lstm_model(max_words=len(tokenizer.word_index)+1, n_classes=len(classes))

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Plot training and validation accuracy
def plot_training_history(history):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['accuracy'], label='train accuracy')
    ax.plot(history.history['val_accuracy'], label='val accuracy')
    ax.set_title('Model Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='upper left')
    plt.show()

plot_training_history(history)

# Save the model and tokenizer
model.save('person_classification_model.keras')
tokenizer_path = 'person_classification_tokenizer'
with open(tokenizer_path, 'wb') as file:
    pickle.dump(tokenizer, file)

# Function to classify a single tweet
def classify_tweet(tweet, model, tokenizer, max_len=50):
    seq = tokenizer.texts_to_sequences([tweet])
    pseq = pad_sequences(seq, maxlen=max_len)
    predictions = model.predict(pseq)
    return np.argmax(predictions)

# Function to classify tweets of a person and plot the distribution
def classify_person(person_name, model_path, tokenizer_path, person_path, classes):
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    
    df = pd.read_csv(person_path)
    classes_count = {cls: 0 for cls in classes}

    for _, row in df.iterrows():
        tweet = row['tweet']
        processed_tweet = process_tweet(tweet)
        code = classify_tweet(processed_tweet, model, tokenizer)
        topic = classes[code]
        classes_count[topic] += 1

    x = classes_count.keys()
    y = classes_count.values()

    plt.figure(figsize=(5, 5))
    plt.title(person_name, fontdict={'fontsize': 20})
    plt.pie(y, labels=x, autopct='%1.1f%%')
    plt.show()

# Classify tweets from different persons
classify_person("Ashraf", 'person_classification_model.keras', 'person_classification_tokenizer', 'ashraf.csv', classes)
classify_person("Salem", 'person_classification_model.keras', 'person_classification_tokenizer', 'salem.csv', classes)
