import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# New class import - hieu
from tensorflow.keras.layers import Bidirectional, Attention, GlobalAveragePooling1D


# Load the pre-split datasets
train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")
test = pd.read_csv("test.csv")

X_train, y_train = train['clean_review'], train['sentiment']
X_val, y_val = val['clean_review'], val['sentiment']
X_test, y_test = test['clean_review'], test['sentiment']

# Encode labels
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_val_enc = encoder.transform(y_val)
y_test_enc = encoder.transform(y_test)

# Tokenize text data
vocab_size = 10000  # maximum number of words
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to have the same length
maxlen = 200  # maximum review length
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=maxlen, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post')

# Build a simple LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen),
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

#new model - hieu
#read the sentence from both sides
              
"""
# new model using BiLSTM + Attention
model = Sequential([

    # Converts each word into a dense vector representation
    Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen),

    # Bidirectional LSTM reads the sequence in both directions
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),

    # Attention layer helps the model focus on the most important words
    Attention(),
    GlobalAveragePooling1D(),

    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
"""
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train_enc,
    validation_data=(X_val_pad, y_val_enc),
    epochs=5,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate the model on the test set
loss, acc = model.evaluate(X_test_pad, y_test_enc)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# Save the model for later use
model.save("sentiment_model.keras")
print("Model saved successfully")

# Save the tokenizer for later prediction
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
    
print("Tokenizer saved successfully!")