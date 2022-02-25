from select import select
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureExtractor:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.features = pd.DataFrame()
    
    def estimate_features(self, k):
        feature_estimator = SelectKBest(f_classif, k=k)
        fit = feature_estimator.fit(self.X, self.y)
        self.features["score"] = fit.scores_
        self.features["feature"] = self.X.columns
        self.features = self.features.sort_values("score", ascending=False)

        return self.features.head(k)["feature"]