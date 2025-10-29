import pickle
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def evaluate_model(model_path, 
                   tokenizer_path="tokenizer.pkl", 
                   test_path="test.csv",
                   current_maxlen=250): # same maxlen as training model
    
    # Kiểm tra các file thiết yếu
    if not all(os.path.exists(p) for p in [model_path, tokenizer_path, test_path]):
        print(f"Lỗi: Không tìm thấy 1 trong các file: {model_path}, {tokenizer_path}, {test_path}")
        return

    print(f"\nĐang đánh giá model: {model_path}")
    print("Loading model và tokenizer...")
    
    # Tải model (không cần custom_objects nữa)
    model = load_model(model_path)
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Tải và chuẩn bị dữ liệu test
    print("Loading test dataset...")
    test = pd.read_csv(test_path)
    X_test, y_test = test["clean_review"], test["sentiment"]

    # Mã hóa nhãn (Tải label_encoder.pkl sẽ tốt hơn, nhưng cách này vẫn chạy)
    encoder = LabelEncoder()
    y_test_enc = encoder.fit_transform(y_test)

    # Chuyển đổi text thành chuỗi với maxlen chính xác
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=current_maxlen, padding='post')

    # Đánh giá
    print("Evaluating model...")
    loss, acc = model.evaluate(X_test_pad, y_test_enc, verbose=0)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Loss: {loss:.4f}")

    # Dự đoán
    y_pred = (model.predict(X_test_pad, verbose=0) > 0.5).astype("int32")

    # In báo cáo chi tiết
    print("\nClassification Report:")
    print(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_enc, y_pred))

# --- CHẠY ĐÁNH GIÁ ---
evaluate_model(model_path="Final.keras")