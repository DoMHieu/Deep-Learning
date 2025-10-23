import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import custom Attention layer
from Attention import Attention

def evaluate_model(model_path, tokenizer_path="tokenizer.pkl", test_path="test.csv"):
    print("Loading model and tokenizer...")
    
    # Load model with custom_objects for Attention
    model = load_model(model_path, custom_objects={"Attention": Attention})
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    print("Loading test dataset...")
    test = pd.read_csv(test_path)
    X_test, y_test = test["clean_review"], test["sentiment"]

    # Encode labels
    encoder = LabelEncoder()
    y_test_enc = encoder.fit_transform(y_test)

    # Convert text to sequences
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=200, padding='post')

    print("Evaluating model...")
    loss, acc = model.evaluate(X_test_pad, y_test_enc, verbose=0)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Loss: {loss:.4f}")

    # Predictions
    y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")

    print("\nClassification Report:")
    print(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_enc, y_pred))

    return acc, loss

acc, loss = evaluate_model(
    model_path="sentiment_model.keras",
    tokenizer_path="tokenizer.pkl",
    test_path="test.csv"
)
"""
evaluate_model(
    model_path="sentiment_model.keras",
    tokenizer_path="tokenizer.pkl",
    test_path="test.csv"
)

evaluate_model(
    model_path="old_model/old_sentiment_model.keras",
    tokenizer_path="tokenizer.pkl",
    test_path="test.csv"
)
"""

#new model with attention
#Test Accuracy: 0.8976
#Test Loss: 0.2485

#new model witout attention
#Test Accuracy: 0.8908
#Test Loss: 0.258

#old model
#Test Accuracy: 0.7584
#Test Loss: 0.5052
