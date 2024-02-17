import logging

import pandas as pd
import mlflow
from zenml import step 
from zenml.client import Client

from src.model import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

exp_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=exp_tracker.name)
def model_trainer(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
) -> RegressorMixin:
    try: 
        model = None

        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            model = model.train(X_train, y_train)
        else:
            raise ValueError(f"Invalid model name: {config.model_name}")
        return model
    except Exception as e:
        logging.error(f'Error in training model: {e}')
        raise e
