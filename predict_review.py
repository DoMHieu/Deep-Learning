import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os

MODEL_PATH = "Final.keras"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
MAX_SEQUENCE_LENGTH = 250

def load_artifacts(model_path, tokenizer_path, label_encoder_path):
    # Check if all required files exist
    if not all(os.path.exists(p) for p in [model_path, tokenizer_path, label_encoder_path]):
        print(f"Error: Missing one or more files: {model_path}, {tokenizer_path}, {label_encoder_path}")
        return None, None, None
        
    print("Loading model and artifacts...")
    model = tf.keras.models.load_model(model_path)
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        
    with open(label_encoder_path, "rb") as f:
        encoder = pickle.load(f)
        
    print("Ready for prediction.")
    return model, tokenizer, encoder

def clean_text(text):
    # Normalize and clean input text
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Zà-ỹ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(review_text, model, tokenizer, encoder):
    # Preprocess the input text
    cleaned_review = clean_text(review_text)
    sequence = tokenizer.texts_to_sequences([cleaned_review])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # Get model prediction
    prediction_score = model.predict(padded_sequence, verbose=0)[0][0]
    predicted_class_index = 1 if prediction_score >= 0.5 else 0
    
    # Decode label using the label encoder
    predicted_label = encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_label, prediction_score

if __name__ == "__main__":
    model, tokenizer, encoder = load_artifacts(MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH)
    
    if model:
        try:
            while True:
                review = input("\nEnter a movie review (or type 'exit' to quit): ")
                if review.lower() == 'exit':
                    break
                    
                label, score = predict_sentiment(review, model, tokenizer, encoder)
                print(f"Sentiment: {label.capitalize()} (Score: {score:.4f})")
                
        except KeyboardInterrupt:
            print("\nExited.")