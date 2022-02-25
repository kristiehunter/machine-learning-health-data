from unicodedata import category
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning, FitFailedWarning
import warnings

warnings.filterwarnings(action="ignore", category=DataConversionWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
warnings.filterwarnings(action="ignore", category=FitFailedWarning)

class MLModel:
    def __init__(self, estimator, parameters, cross_validation):
        self.estimator = estimator
        self.parameters = parameters
        self.cv = cross_validation

    def calculate_gridsearch(self, X, Y):
        grid_result = GridSearchCV(self.estimator, self.parameters, cv=self.cv)
        grid_result.fit(X, Y)

        return grid_result