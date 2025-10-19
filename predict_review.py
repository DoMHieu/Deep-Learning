import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Load the trained model and tokenizer
model = tf.keras.models.load_model("sentiment_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define a text cleaning function (same as during training)
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = text.lower()
    return text

# Get user input
review = input("Enter a movie review: ")

# Preprocess and tokenize the input text
cleaned = clean_text(review)
seq = tokenizer.texts_to_sequences([cleaned])
padded = pad_sequences(seq, maxlen=200)

# Make prediction
pred = model.predict(padded)[0][0]
if pred >= 0.5:
    print(f"Sentiment: Positive ({pred:.2f})")
else:
    print(f"Sentiment: Negative ({pred:.2f})")
