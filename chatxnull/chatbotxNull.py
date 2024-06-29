"""This chatbot have the ability to read and train on PDF data using an RNN, we need to perform the following steps:

1. Extract text from PDF files.
2. Preprocess the text data.
3. Create an RNN model for training.
4. Train the model using the preprocessed data.
5. Save the model and related components.
6. Develop a chatbot interface to interact with the trained model."""

"""Here's a complete implementation:"""

### Step 1: Extract text from PDF files

"""We'll use the `PyMuPDF` library to extract text from PDF files."""

# install PyMuPDF
"""!pip install pymupdf"""

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

### Step 2: Preprocess the text data

"""We'll use NLTK for text preprocessing."""


import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word.lower()) for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens


### Step 3: Create an RNN model for training


from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, LSTM, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Parameters
vocab_size = 5000
max_len = 100

# Tokenizer
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(["sample text for fitting"])

def create_rnn_model(input_dim, output_dim, input_length):
    model = Sequential()
    model.add(Input(shape=(input_length,)))
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(len(classes), activation='softmax'))
    return model


### Step 4: Train the model


def train_model(text_data, labels, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(text_data)
    X = pad_sequences(sequences, maxlen=max_len)
    y = labels  # Ensure labels are one-hot encoded
    
    model = create_rnn_model(vocab_size, 128, max_len)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32)
    return model


### Step 5: Save the model and related components


import pickle

def save_model(model, tokenizer, path_prefix):
    model.save(f"{path_prefix}_model.keras")
    with open(f"{path_prefix}_tokenizer.pkl", 'wb') as file:
        pickle.dump(tokenizer, file)

# Example usage
# save_model(model, tokenizer, "chatbot")


### Step 6: Develop a chatbot interface


def chatbot_interface(model_path, tokenizer_path, intents_path):
    from keras.models import load_model

    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)

    with open(intents_path) as file:
        data = json.load(file)

    def get_response(text):
        sequence = tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(sequence, maxlen=max_len)
        pred = model.predict(padded_seq)
        class_label = np.argmax(pred)
        return data['intents'][class_label]['responses'][0]  # assuming there's always a response

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = get_response(user_input)
        print(f"Bot: {response}")

# Example usage
# chatbot_interface("chatbot_model.keras", "chatbot_tokenizer.pkl", "intents.json")


### Example Main Script


import json

with open('intents.json') as file:
    data = json.load(file)

# Step 1: Extract and preprocess text
pdf_text = extract_text_from_pdf('sample.pdf')
processed_text = preprocess_text(pdf_text)

# Step 2: Prepare training data
tokenizer.fit_on_texts(processed_text)
sequences = tokenizer.texts_to_sequences(processed_text)
X = pad_sequences(sequences, maxlen=max_len)

labels = np.array([0] * len(X))  # Dummy labels, replace with actual labels

# Step 3: Train and save model
model = train_model(processed_text, labels, tokenizer, max_len)
save_model(model, tokenizer, "chatbot")

# Step 4: Chatbot interface
chatbot_interface("chatbot_model.keras", "chatbot_tokenizer.pkl", "intents.json")


