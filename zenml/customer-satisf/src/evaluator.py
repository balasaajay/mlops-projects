import logging

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluator(ABC):

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class MSE(Evaluator):

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info(f'Evaluation for MSE started...')
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f'Evaluation for MSE done!')
            return mse
        except Exception as e:
            logging.error(f'Error in evaluation for MSE: {e}')
            raise e

class R2Score(Evaluator):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info(f'Evaluation for R2Score started...')
            r2s = r2_score(y_true, y_pred)
            logging.info(f'Evaluation for R2Score done!')
            return r2s
        except Exception as e:
            logging.error(f'Error in evaluation for R2Score: {e}')
            raise e

class RMSE(Evaluator):

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info(f'Evaluation for RMSE started...')
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f'Evaluation for RMSE done!')
            return rmse
        except Exception as e:
            logging.error(f'Error in evaluation for RMSE: {e}')
            raise e