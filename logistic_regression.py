
# 필요한 모듈과 데이터 불러오기
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings
import pickle

filterwarnings('ignore')
cancer = pd.read_csv('./data/dtm_matrix_origin.csv', encoding='cp949')
columns = cancer.columns
print(len(columns))
df = pd.DataFrame(cancer, columns=columns)
print(df.shape)

# 데이터를 훈련셋과 테스트셋으로 나누기
X = cancer.iloc[:,:-1]
y = cancer.iloc[:,-1]
print(y, columns[-1])
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    test_size=0.2)
print(df.shape)
print(X.shape, y.shape)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
model_lr = LogisticRegression()
model_tr = DecisionTreeClassifier()
model_rf = RandomForestClassifier()

model_lr.fit(X_train, y_train)
model_tr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

f = open('./classification_for_api/model/model_lr.pkl', 'wb')
pickle.dump(model_lr,f)
f.close()


# 평가
print(model_lr.score(X_train, y_train), model_lr.score(X_test, y_test))
print(model_tr.score(X_train, y_train), model_tr.score(X_test, y_test))
print(model_rf.score(X_train, y_train), model_rf.score(X_test, y_test))
print(X_train.shape)
idx = 7
print(model_lr.predict(X_train.iloc[idx,:].values.reshape(1,-1)), y_train[idx])
