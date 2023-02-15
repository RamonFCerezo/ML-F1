from sklearn.ensemble import RandomForestClassifier
import pandas as pd

rnd_clf = RandomForestClassifier(n_estimators=200,
                                 max_features = 5,
                                 max_depth = 5,
                                 max_leaf_nodes=8,
                                 random_state=42)

X_train = pd.read_csv('ML-F1/src/data/X_train.csv', index_col ='ResultId')
Y_train = pd.read_csv('ML-F1/src/data/y_train.csv')

Y_train.drop('ResultId', axis=1, inplace=True)

rnd_clf.fit(X_train, Y_train)

import pickle

with open('ML-F1/src/model/modelotrain.model', "wb") as archivo_salida:
    pickle.dump(rnd_clf, archivo_salida)