import yaml
import time
import json

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from src.preprocessor.preprocessor import DataPreProcessor
from src.feature_extractor.feature_extractor import FeatureExtractor
from src.model.model import MLModel

DATA_FOLDER = "./data/"

with open('config.yaml') as f:
    CONFIG_FILE = yaml.load(f, Loader=yaml.FullLoader)

TARGET = CONFIG_FILE["TARGET"]
TARGET_MIN = CONFIG_FILE["TARGET_MIN"]
TARGET_MAX = CONFIG_FILE["TARGET_MAX"]

with open("config/model_parameters.json") as MODEL_PARAM_FILE:
    MODEL_PARAMS = json.load(MODEL_PARAM_FILE)

if __name__ == "__main__":
    print("===== Program Start =====")

    try:
        print("Importing masterdata.")
        master_data = pd.read_csv(f"{DATA_FOLDER}/masterdata.csv")
    except FileNotFoundError:
        print("masterdata file is missing.  Please upload the data first.")
    else:
        null_methods = ["interpolate", "drop", "fill_mean", "fill_previous"]
        scaler_methods = ["robust", "max_abs", "min_max"]
        models = [
            MLModel(MLPClassifier(), MODEL_PARAMS["mlp_params"], 5),
            MLModel(GaussianNB(), MODEL_PARAMS["nb_params"], 5),
            MLModel(KNeighborsClassifier(), MODEL_PARAMS["knn_params"], 5),
            MLModel(DecisionTreeClassifier(), MODEL_PARAMS["tree_params"], 5),
            MLModel(SVC(), MODEL_PARAMS["svc_params"], 5),
            MLModel(LinearDiscriminantAnalysis(), MODEL_PARAMS["lda_params"], 5),
            MLModel(LogisticRegression(), MODEL_PARAMS["lr_params"], 5)
        ]
        k_estimates = [2, 3, 5, 8, 10]

        best_score = 0
        best_combination = {
            "null_method": None,
            "scale_method": None,
            "k": None,
            "m": None,
            "params": None,
            "features": None
        }

        for null_method in null_methods:
            for scale_method in scaler_methods:
                print(f"Testing out {null_method} replace null method with {scale_method} scaling.")
                preprocess = DataPreProcessor(master_data, TARGET, null_method)
                preprocess.scale_values(scale_method)
                preprocess.categorize_target(TARGET, [TARGET_MIN, 0, TARGET_MAX], [0, 1])
                
                train_X, train_y, test_X, test_y = preprocess.split_data(0.2)

                for k in k_estimates:
                    features = FeatureExtractor(train_X, train_y)
                    k_features = features.estimate_features(k)
                            
                    train_X_features = train_X[k_features]
                    test_X_features = test_X[k_features]

                    for m in models:
                        start_time = time.time()
                        trained_model = m.calculate_gridsearch(train_X, train_y[TARGET])
                        end_time = time.time()

                        print(f"Time to train model {str(m.estimator)} with {k} features: {(end_time - start_time)/60:.2f} minutes")
                        print(f"Score: {(trained_model.best_score_)*100:.2f}%")

                        if trained_model.best_score_ > best_score:
                            best_score = trained_model.best_score_
                            best_combination["k"] = k
                            best_combination["m"] = trained_model.best_estimator_
                            best_combination["params"] = trained_model.best_params_
                            best_combination["null_method"] = null_method
                            best_combination["scale_method"] = scale_method
                            best_combination["features"] = k_features
        
        print("\n--- BEST OVERALL RESULTS ---\n")
        print(f"Best Score: {best_score}")
        print(f"""Best Combination:
        \t Null Method - {best_combination["null_method"]}
        \t Scaling Method - {best_combination["scale_method"]}
        \t Number of Features - {best_combination["k"]}
        \t Model - {str(best_combination["m"])}
        \t Model Parameters - {best_combination["params"]}
        """)

        print("\nTraining the full data training set using the combination.")
        preprocess = DataPreProcessor(master_data, TARGET, best_combination["null_method"])
        preprocess.scale_values(best_combination["scale_method"])
        preprocess.categorize_target(TARGET, [TARGET_MIN, 0, TARGET_MAX], [0, 1])
        train_X, train_y, test_X, test_y = preprocess.split_data(0.2)
                            
        train_X_features = train_X[best_combination["features"]]
        test_X_features = test_X[best_combination["features"]]

        estimator = best_combination["m"]
        estimator.fit(train_X_features, train_y)
        training_predictions = estimator.predict(train_X_features)
        testing_predictions = estimator.predict(test_X_features)

        training_predictions_data = train_X_features.copy()
        testing_predictions_data = test_X_features.copy()

        training_predictions_data["prediction"] = training_predictions
        testing_predictions_data["prediction"] = testing_predictions

        training_predictions_data["actual_target"] = train_y[TARGET]
        testing_predictions_data["actual_target"] = test_y[TARGET]

        training_predictions_data.to_csv(DATA_FOLDER + TARGET + "_training_predictions.csv", index=False, header=True)
        testing_predictions_data.to_csv(DATA_FOLDER + TARGET + "_testing_predictions.csv", index=False, header=True)

        best_combination["m"] = str(best_combination["m"])
        best_combination["features"] = best_combination["features"].tolist()

        with open(DATA_FOLDER + TARGET + '_best_estimators.json', 'w') as f:
            json.dump(best_combination, f)