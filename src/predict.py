import pickle
import pandas as pd

with open('model/modelo.model', "rb") as archivo_entrada:
    pipeline_importada = pickle.load(archivo_entrada)
 
X_test = pd.read_csv('data/x_test.csv', index_col ='ResultId')

predictions = pipeline_importada.predict(X_test)

pd.DataFrame(predictions, columns=['predictions']).to_csv('data/prediction.csv')