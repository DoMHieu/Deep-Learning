import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os

# --- Cấu hình ---
MODEL_PATH = "Final.keras"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
MAX_SEQUENCE_LENGTH = 250 # Phải khớp với model đã huấn luyện (là 250)

def load_artifacts(model_path, tokenizer_path, label_encoder_path):
    """
    Tải model, tokenizer, và label encoder.
    """
    if not all(os.path.exists(p) for p in [model_path, tokenizer_path, label_encoder_path]):
        print(f"Lỗi: Không tìm thấy 1 trong các file: {model_path}, {tokenizer_path}, {label_encoder_path}")
        return None, None, None
        
    print("Đang tải model và các artifacts...")
    model = tf.keras.models.load_model(model_path)
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        
    with open(label_encoder_path, "rb") as f:
        encoder = pickle.load(f)
        
    print("Sẵn sàng dự đoán.")
    return model, tokenizer, encoder

def clean_text(text):
    """
    Hàm làm sạch văn bản, PHẢI GIỐNG HỆT lúc huấn luyện.
    (Phiên bản này loại bỏ HTML, chỉ giữ lại chữ cái và khoảng trắng)
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)           # Loại bỏ thẻ HTML
    text = re.sub(r"[^a-zA-Z\s]", '', text)   # Chỉ giữ lại chữ cái và khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()    # Xóa khoảng trắng thừa
    return text

def predict_sentiment(review_text, model, tokenizer, encoder):
    """
    Làm sạch, tokenize, pad và dự đoán sentiment của một review.
    """
    # Xử lý văn bản
    cleaned_review = clean_text(review_text)
    
    # Chuyển thành chuỗi số
    sequence = tokenizer.texts_to_sequences([cleaned_review])
    
    # Pad chuỗi
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # Dự đoán (cho ra một số từ 0 đến 1)
    prediction_score = model.predict(padded_sequence, verbose=0)[0][0]
    
    # Chuyển điểm số thành 0 hoặc 1
    predicted_class_index = 1 if prediction_score >= 0.5 else 0
    
    # Dùng encoder để lấy nhãn (ví dụ: "positive" hoặc "negative")
    predicted_label = encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_label, prediction_score

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    model, tokenizer, encoder = load_artifacts(MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH)
    
    if model:
        try:
            while True:
                review = input("\nEnter a movie review (hoặc gõ 'exit' để thoát): ")
                if review.lower() == 'exit':
                    break
                    
                label, score = predict_sentiment(review, model, tokenizer, encoder)
                
                print(f"Sentiment: {label.capitalize()} (Score: {score:.4f})")
                
        except KeyboardInterrupt:
            print("\nĐã thoát.")