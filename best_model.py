import yaml

import json

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Take this out?
from sklearn.linear_model import LogisticRegression

from src.preprocessor.preprocessor import DataPreProcessor

DATA_FOLDER = "./data/"

with open('config.yaml') as f:
    CONFIG_FILE = yaml.load(f, Loader=yaml.FullLoader)

TARGET = CONFIG_FILE["TARGET"]
TARGET_MIN = CONFIG_FILE["TARGET_MIN"]
TARGET_MAX = CONFIG_FILE["TARGET_MAX"]

if __name__ == "__main__":
    print("===== Program Start =====\n")

    try:
        print("Retrieving best model file.")
        with open(f"{DATA_FOLDER}/{TARGET}_best_estimators.json") as MODEL_PARAM_FILE:
            best_combination = json.load(MODEL_PARAM_FILE)
    except FileNotFoundError:
        print("Best estimator file doesn't exist yet, please run main.py first.")
    else:
        print("Importing masterdata.")
        try:
            master_data = pd.read_csv(f"{DATA_FOLDER}/masterdata.csv")
        except FileNotFoundError:
            print("masterdata file is missing.  Please upload the data first.")
        else:
            print(f"Preprocessing data with {best_combination['null_method']} null method and {best_combination['scale_method']} scale method.")
            preprocess = DataPreProcessor(master_data, TARGET, best_combination["null_method"])
            preprocess.scale_values(best_combination["scale_method"])
            preprocess.categorize_target(TARGET, [TARGET_MIN, 0, TARGET_MAX], [0, 1])
            train_X, train_y, test_X, test_y = preprocess.split_data(0.2)

            train_X_features = train_X[best_combination["features"]]
            test_X_features = test_X[best_combination["features"]]

            print("Creating estimator model.")
            estimator = best_combination["m"]
            try:
                # TODO: With Python 3 upgrade, replace with case matching
                if estimator == "MLPClassifier()":
                    estimator = MLPClassifier(**best_combination["params"])
                elif estimator == "KNeighborsClassifier()":
                    estimator = KNeighborsClassifier(**best_combination["params"])
                elif estimator == "SVC()":
                    estimator = SVC(**best_combination["params"])
                elif estimator == "DecisionTreeClassifier()":
                    estimator = DecisionTreeClassifier(**best_combination["params"])
                elif estimator == "GaussianNB()":
                    estimator = GaussianNB(**best_combination["params"])
                elif estimator == "LinearDiscriminantAnalysis()":
                    estimator = LinearDiscriminantAnalysis(**best_combination["params"])
                elif estimator == "LogisticRegression()":
                    estimator = LogisticRegression(**best_combination["params"])
                else:
                    raise ValueError("Model not found.")
            except ValueError:
                print("Model found in the JSON file does not match the accepted options.")
            else:
                print("Training model.")
                estimator.fit(train_X_features, train_y)

                print("Predicting and saving data.")
                training_predictions = estimator.predict(train_X_features)
                testing_predictions = estimator.predict(test_X_features)

                training_predictions_data = train_X_features.copy()
                testing_predictions_data = test_X_features.copy()

                training_predictions_data["prediction"] = training_predictions
                testing_predictions_data["prediction"] = testing_predictions

                training_predictions_data["actual_target"] = train_y[TARGET]
                testing_predictions_data["actual_target"] = test_y[TARGET]

                training_predictions_data.to_csv(DATA_FOLDER + TARGET + "_training_predictions_best_model.csv", index=False, header=True)
                testing_predictions_data.to_csv(DATA_FOLDER + TARGET + "_testing_predictions_best_model.csv", index=False, header=True)