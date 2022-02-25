import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

class DataPreProcessor:
    def __init__(self, data, target, null_method):
        
        try:
            data = self.fill_null_values(data, null_method)
        except ValueError:
            print("Method was not recognized; data stays as is.")
        finally:
            self.y = data[target]
            self.X = data.drop([target], axis=1)

    def fill_null_values(self, data, method):

        if method == "interpolate":
            return data.interpolate(method ='linear', limit_direction ='forward')
        elif method == "drop":
            return data.dropna()
        elif method == "fill_mean":
            return data.fillna(data.mean())
        elif method == "fill_previous":
            return data.fillna(method ='pad')
        else:
            raise ValueError(f"Method {method} is not recognized.")

    def scale_values(self, scale_method):
        
        if scale_method == "robust":
            scaler = RobustScaler()
        elif scale_method == "max_abs":
            scaler = MaxAbsScaler()
        elif scale_method == "min_max":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)

    def categorize_target(self, target_name, bins, labels):
        
        categorized = pd.cut(self.y, bins, labels=labels)
        self.y = pd.DataFrame({target_name: categorized})

    def split_data(self, test_size):
        splitter = StratifiedShuffleSplit(n_splits = 1, test_size = test_size)
        data = []

        for set1, set2 in splitter.split(self.X, self.y):
            data += [self.X.iloc[set1]]
            data += [self.y.iloc[set1]]
            data += [self.X.iloc[set2]]
            data += [self.y.iloc[set2]]
        
        return data[0], data[1], data[2], data[3]

