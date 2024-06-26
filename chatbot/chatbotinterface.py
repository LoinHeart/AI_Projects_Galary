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
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens ]
    
    stop_words = stopwords.words('english')
    
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens


def bag_of_words(text, vocab): 
    tokens = clean_text(text)
    bow = np.zeros(len(vocab))
    for w in tokens: 
        for idx, word in enumerate(vocab):
            if word == w: 
                bow[idx] = 1
    return bow.reshape(1, -1)  # Reshape to a 2D array

def pred_class(text, model, vocab, classes): 
    
    bow = bag_of_words(text, vocab)
    predictions = model.predict([bow], verbose=0)[0]
    
    max_index=np.argmax(predictions)
    
    return classes[max_index]

def get_response(pred_cl, intents_json): 
    list_of_intents = intents_json["intents"]
    for x in list_of_intents: 
        if x["class"] == pred_cl:
            result = random.choice(x["responses"])
            break
    return result
def chatbot(intents_path, model_path, vocab_path, classes_path): 
    with open(intents_path) as file:
        data = json.load(file)

    model = load_model(model_path)

    with open(vocab_path, 'rb') as file:
        vocab = pickle.load(file)

    with open(classes_path, 'rb') as file:
        classes = pickle.load(file)

    message = input("Start Chatting : ")
    while len(message)>0:
        message= message.lower()
        pred_cl = pred_class(message, model, vocab, classes)
        result = get_response(pred_cl, data)
        print(result)
        message = input("You: ")
chatbot(r'Q:\Ai Courses\AI Hsoub\DeepLearning\Course Project\Jp_notebook\(15) بناء بوت محادثة\X-03\intents.json', r'Q:\Ai Courses\AI Hsoub\DeepLearning\Course Project\Jp_notebook\(15) بناء بوت محادثة\X-03\chatbot_model.keras', r'Q:\Ai Courses\AI Hsoub\DeepLearning\Course Project\Jp_notebook\(15) بناء بوت محادثة\X-03\chatbot_vocab', r'Q:\Ai Courses\AI Hsoub\DeepLearning\Course Project\Jp_notebook\(15) بناء بوت محادثة\X-03\chatbot_classes')