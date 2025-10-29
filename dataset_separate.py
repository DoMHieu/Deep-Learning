import pandas as pd
from sklearn.model_selection import train_test_split

#Reading processed 
data = pd.read_csv("IMDB_clean.csv")

#Traing set 80%,    Temp set 20%
X_train, X_temp, y_train, y_temp = train_test_split(
    data['clean_review'], data['sentiment'],
    test_size=0.2, random_state=42
)

#Temp set: Test set 10%,   Valid set 10%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5, random_state=42
)

#Create DataFrame to store
train_df = pd.DataFrame({'clean_review': X_train, 'sentiment': y_train})
val_df = pd.DataFrame({'clean_review': X_val, 'sentiment': y_val})
test_df = pd.DataFrame({'clean_review': X_test, 'sentiment': y_test})

#Extracting to csv file
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))
