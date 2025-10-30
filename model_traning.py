import os
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Reproducibility setup
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

vocab_size = 30000
maxlen = 250
embedding_dim = 300
fasttext_path = "crawl-300d-2M-subword.vec" 

train_csv = "train.csv"
val_csv = "val.csv"
test_csv = "test.csv"

# Load datasets
print("Loading separated datasets...")
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

X_train, y_train = train_df['clean_review'], train_df['sentiment']
X_val, y_val = val_df['clean_review'], val_df['sentiment']
X_test, y_test = test_df['clean_review'], test_df['sentiment']

print(f"Loaded {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples.")

# Encode labels
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_val_enc = encoder.transform(y_val)
y_test_enc = encoder.transform(y_test)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Tokenization and padding
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=maxlen, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Load pretrained FastText embeddings
def load_fasttext_embeddings(file_path, embedding_dim, word_index, num_words):
    print(f"Loading pre-trained embeddings from: {file_path}")
    embeddings_index = {}
    try:
        with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
            next(f)
            for line in f:
                values = line.rstrip().split(" ")
                word = values[0]
                if len(values) == embedding_dim + 1:
                    try:
                        coefs = np.asarray(values[1:], dtype='float32')
                        embeddings_index[word] = coefs
                    except ValueError:
                        pass
    except FileNotFoundError:
        print("\n" + "="*60)
        print(f"ERROR: Missing embedding file at: {file_path}")
        print("Please download 'crawl-300d-2M-subword.vec.zip' from:")
        print("https://fasttext.cc/docs/en/english-vectors.html")
        print("Unzip and place the .vec file in the same folder as this script.")
        print("="*60 + "\n")
        exit()

    embedding_matrix = np.zeros((num_words, embedding_dim))
    words_found = 0
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            words_found += 1
    print(f"Found {words_found} / {num_words} words in pre-trained file.")
    return embedding_matrix

word_index = tokenizer.word_index
num_words = min(vocab_size, len(word_index) + 1)
embedding_matrix = load_fasttext_embeddings(fasttext_path, embedding_dim, word_index, num_words)

# Build GRU model with pooling layers
def build_model(num_words, embedding_dim, embedding_matrix, maxlen):
    inputs = Input(shape=(maxlen,))
    x = Embedding(input_dim=num_words,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=maxlen,
                  trainable=False)(inputs)  # Phase 1: freeze embeddings
    
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.2))(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

# Phase 1: Train with frozen embeddings
model = build_model(num_words, embedding_dim, embedding_matrix, maxlen)
optimizer_p1 = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5)
model.compile(optimizer=optimizer_p1,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

checkpoint_p1_path = "Phase1.keras"
callbacks_p1 = [
    ModelCheckpoint(checkpoint_p1_path, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

print("--- STARTING PHASE 1 (FROZEN EMBEDDINGS) ---")
model.fit(
    X_train_pad, y_train_enc,
    validation_data=(X_val_pad, y_val_enc),
    epochs=15,
    batch_size=64,
    callbacks=callbacks_p1,
    verbose=1
)

# Phase 2: Fine-tuning embeddings
print("\n--- STARTING PHASE 2 (FINE-TUNING EMBEDDINGS) ---")
model.load_weights(checkpoint_p1_path)
model.layers[1].trainable = True

optimizer_p2 = tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-6)
model.compile(optimizer=optimizer_p2,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

checkpoint_p2_path = "Phase2.keras"
callbacks_p2 = [
    ModelCheckpoint(checkpoint_p2_path, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
]

model.fit(
    X_train_pad, y_train_enc,
    validation_data=(X_val_pad, y_val_enc),
    epochs=10,
    batch_size=32,
    callbacks=callbacks_p2,
    verbose=1
)

# Evaluate the best model on test data
print("Loading best model (Phase 2) for test evaluation...")
best_model_path = checkpoint_p2_path if os.path.exists(checkpoint_p2_path) else checkpoint_p1_path
model.load_weights(best_model_path)

test_loss, test_acc = model.evaluate(X_test_pad, y_test_enc, verbose=1)
print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")

model.save("Final.keras")
print("Model and artifacts have been saved.")