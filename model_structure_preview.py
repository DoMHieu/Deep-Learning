from tensorflow.keras.models import load_model
model = load_model(r"D:\DeepLearning\sentiment_model.keras")
# muốn chạy nhớ chỉnh lại link model
# print out the structure of the model
model.summary()
