# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from xgboost import XGBClassifier as XGB
import pandas as pd
import pickle

df = pd.read_csv('./heart.csv')

cate = ['Sex','ExerciseAngina','RestingECG','ChestPainType','ST_Slope']

for i in cate:
    LE = preprocessing.LabelEncoder()
    df[i] = LE.fit_transform(df[i])

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

final_model = XGB(n_estimators=400, max_depth=5)
final_model.fit(X,Y)

pickle.dump(final_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model)