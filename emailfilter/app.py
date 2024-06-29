import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset from 'emails.csv'
df = pd.read_csv('emails.csv')

# Preprocess the text data
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stop words
    stop_words = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stem the words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into a preprocessed text
    preprocessed_text = ' '.join(tokens)
    
    # Remove URLs and digits
    preprocessed_text = re.sub(r'http\S+|www\S+', '', preprocessed_text)
    preprocessed_text = re.sub(r'\d+', '', preprocessed_text)
    
    return preprocessed_text

# Apply preprocessing to the 'Message' column
df['processed_Message'] = df['Message'].apply(preprocess_text)

# Create a TF-IDF vectorizer
max_features = 100
tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
tfidf_vectors = tfidf_vectorizer.fit_transform(df['processed_Message'])

# Split the data into training and testing sets
X = tfidf_vectors.toarray()
y = df['Spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Build a neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test data
y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2))

# Example: Predict spam/ham for a new message
message = "call to get free prize one million dollars"
processed_message = preprocess_text(message)
vector = tfidf_vectorizer.transform([processed_message])
vector_dense = vector.toarray()
y_pred_prob = model.predict(vector_dense)
y_pred = np.round(y_pred_prob)
print("Predicted class:", "Spam" if y_pred[0][0] == 1 else "Ham")
