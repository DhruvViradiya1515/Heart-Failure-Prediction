import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Age	Sex	ChestPainType	RestingBP	Cholesterol	FastingBS	RestingECG	MaxHR	ExerciseAngina	Oldpeak	ST_Slope
    #  ['Sex','ExerciseAngina','RestingECG','ChestPainType','ST_Slope']
    features = list(request.form.values())
    final_features = []
    possible_features = [['f','m'],['asy','ata','nap','ta'],['lvh','normal','st'],['n','y'],['down','flat','up']]
    cate_features = [1, 2, 6, 8, 10]
    for i, val in enumerate(features):
        if i in cate_features:
            ix = cate_features.index(i)
            final_features.append(possible_features[ix].index(val.lower()))
        else:
            final_features.append(float(val))

    ytest = pd.DataFrame(np.reshape(final_features, (1,11)))
    ytest.columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS','RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

    prediction = model.predict_proba(ytest)
    output = float(prediction[0][1])

    return render_template('index.html', prediction_text=f'Heart Failure Probability : {output*100:.4f} %') 

if __name__ == "__main__":
    app.run(debug=True)