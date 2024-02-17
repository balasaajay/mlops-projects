import logging

from abc import ABC, abstractmethod

import pandas as pd

from sklearn.linear_model import LinearRegression

class Model(ABC):

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

class LinearRegressionModel(Model):
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs):
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f'Error in training LinearRegressionModel: {e}')
            raise e
