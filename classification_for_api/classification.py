import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Vectorizer
df = pd.read_csv('./data/발라드.csv', encoding='cp949')
vectorizer = pickle.load(open("./model/count_vectorizer.pkl", 'rb'))

# 가사는 배열로 넣어야 함.
lyrics = [ df.loc[0,'lyrics'] ]
vect = vectorizer.transform([df.loc[0,'lyrics']])
temp_df = pd.DataFrame(vect.A, columns=vectorizer.get_feature_names_out())

# Logistic Regression
model_lr = pickle.load(open("./model/model_lr.pkl", 'rb'))
print(temp_df.values)
print(model_lr.predict(temp_df.values))