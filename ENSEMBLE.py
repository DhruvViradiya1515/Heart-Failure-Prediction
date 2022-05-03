import numpy as np
from xgboost import XGBClassifier as XGB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
import pandas as pd


class ENSEMBLE():

    def __init__(self):
        self.models = [XGB(n_estimators=400, max_depth=5),
                       RandomForestClassifier(random_state=0),
                       LGBMClassifier(),
                       GradientBoostingClassifier(),
                       GaussianNB()]
    
    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        for weak_learner in self.models:
            weak_learner.fit(X, Y)


    def predict(self, X_test):
        self.X_test = X_test
        ypreds = []
        for weak_learner in self.models:
            ypreds.append(weak_learner.predict(X_test))

        ypreds = np.array(ypreds)
        final_preds = []
        for arr in ypreds.T:
            vals,counts = np.unique(arr, return_counts=True)
            index = np.argmax(counts)
            final_preds.append(vals[index])

        final_preds = np.array(final_preds)

        return final_preds


    def predict_proba(self, X_test):
        self.X_test = X_test
        yprobas1 = []
        yprobas0 = []
        for weak_learner in self.models:
            yprobas1.append(np.array(weak_learner.predict_proba(X_test))[:,1])
            yprobas0.append(np.array(weak_learner.predict_proba(X_test))[:,0])

        yprobas0 = np.array(yprobas0)
        yprobas1 = np.array(yprobas1)
        proba1 = []
        proba0 = []
        for i in range(len(yprobas0[0])):
            proba1.append(np.mean(yprobas1[:,i]))
            proba0.append(np.mean(yprobas0[:,i]))

        proba1 = np.array(proba1)
        proba0 = np.array(proba0)
        final_proba = np.array([proba0, proba1])

        return final_proba.T