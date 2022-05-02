import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from xgboost import XGBClassifier as XGB
import pandas as pd
import pickle


df = pd.read_csv('./heart.csv')

cate = ['Sex','ExerciseAngina','RestingECG','ChestPainType','ST_Slope']

# print(df.head().to_markdown())

# for i in cate:
#     print(i, df[i].value_counts())
#     LE = preprocessing.LabelEncoder()
#     df[i] = LE.fit_transform(df[i])

# print(df.head().to_markdown())

features = df.iloc[0,:-1]
print('raw_features',features)
final_features = []
possible_features = [['f','m'],['asy','ata','nap','ta'],['lvh','normal','st'],['n','y'],['down','flat','up']]
cate_features = [1, 2, 6, 8, 10]
for i, val in enumerate(features):
    if i in cate_features:
        ix = cate_features.index(i)
        final_features.append(possible_features[ix].index(val.lower()))
    else:
        final_features.append(val)

print('final features',final_features)