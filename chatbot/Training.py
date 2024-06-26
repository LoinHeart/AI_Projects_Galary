from X_03 import *
# Compile the Neural Network model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Neural Network
model.fit(X, y, epochs=1000)

# Save the trained model if needed
model.save('chatbot_model.keras')
vocab_path = 'chatbot_vocab'
with open(vocab_path, 'wb') as file:
    pickle.dump(vocab, file)

classes_path = 'chatbot_classes'
with open(classes_path, 'wb') as file:
    pickle.dump(classes, file)
