import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

data = pd.read_csv('IMDB Dataset.csv')

def clean_text(text):
    text = text.lower()      #convert to lower case
    text = re.sub(r'<.*?>', '', text)     #remove HTML tag
    text = re.sub(r'[^a-zA-Zà-ỹ\s]', '', text)    #remove special characters
    text = re.sub(r'\s+', ' ', text).strip()   #remove space
    return text

data['clean_review'] = data['review'].apply(clean_text)

print(data.head())

data.to_csv('IMDB_clean.csv', index=False)

