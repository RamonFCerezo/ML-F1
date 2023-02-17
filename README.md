# ML-F1
Prediction model for F1 races

### 1. Introduction

Formula 1 is a very complex sport, where results depend on a lot of factors, from the engineering behind the car, to the weather forecast. Of course, we will admit that driver's ability plays an important role in order to know how a race is going to finish, although some haters doubt it. Every sport has a certain part of uncertainty, which makes them funnier, but in this project we are going to minimize that, until we get the better predictions possible.

**Objectives**

- Create a **ML classifier model** to predict the final results of a race based on previous features.
- Understand the different features that have influence in the final results, and emphasize in them.

### 2. Structure

- Data: The original database we will use comes from the API: 'http://ergast.com/mrd/'. Nevertheless, someone already did the work for us, because we can obtain the same dataset from Kaggle: 'https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020', divided in 13 csv files:
    - circuits.csv
    - constructor_results.csv
    - constructor_standings.csv
    - constructors.csv
    - driver_standings.csv
    - drivers.csv
    - lap_times.csv
    - pit_stops.csv
    - qualifying.csv
    - races.csv
    - results.csv
    - status.csv

- Additional files in data: X_train.csv, x_test.csv, y_test.csv, y_train.csv and predictions.csv (this is the path where all new files are sent)

- Model: 4 models, all of them RandomForestClassifier
    - Main
    - Additionals (different target and features)

- Notebooks
- memoria.ipynb: summary of every process done
- predict.py (only for the main model)
- train.py (only for the main model)

### 3. Steps

- Data preprocessing (database ellaboration, feature selection, feature engineering, etc)
- Attempts with different ML models (pipeline including SVC, RandomForestClassifier, Xgboost, KNN, logistic regression)
- Understanding the results (confusion matrix, feature importance, accuracy score, best params, etc)
- Conclusions
- Solutions, ways to improve our models

### 4. Conclusions

This ML model based on the last 6 seasons works for the last races of every season with promising results. Nevertheless, there are also ways to improve it:

- More accurate DNF index that could predict this feature would be an interesting solution to imitate the good results that the real DNF column shows. It would be necessary, for example, check how dangerous is the first lap in each circuit, how many races have the different parts of the car taken in (to check how far are they to break), which drivers are more likely to an accident, etc.

- Find a way so it chooses the 10 drivers with more options to be 1, and the 10 drivers with more options to be 0, not just when a driver is above 0.5 in the predict proba.

- More data: weather as a randomizer, but also to check the laptimes with and without weather, different strategies, pit-stops, Safety-cars deployed, success in the first lap, etc.

- Find a way that allows our model to choose the last positions.