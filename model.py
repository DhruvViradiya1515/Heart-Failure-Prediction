from sklearn import preprocessing
import pandas as pd
import pickle
from ENSEMBLE import ENSEMBLE

df = pd.read_csv('./heart.csv')

cate = ['Sex', 'ExerciseAngina', 'RestingECG', 'ChestPainType', 'ST_Slope']

for i in cate:
    LE = preprocessing.LabelEncoder()
    df[i] = LE.fit_transform(df[i])

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

final_model = ENSEMBLE()
final_model.fit(X, Y)

pickle.dump(final_model, open('./model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('./model.pkl', 'rb'))
print(model)