import logging

import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.evaluator import MSE, R2Score, RMSE
from sklearn.base import RegressorMixin

import mlflow
from zenml.client import Client

exp_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=exp_tracker.name)
def evalualte_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "R2Score"],
    Annotated[float, "RMSE"],
]:
    try:

        pred = model.predict(X_test)

        mse_obj = MSE()
        mse = mse_obj.evaluate(y_test, pred)
        mlflow.log_metric("mse", mse)
        
        r2s_obj = R2Score()
        r2s = r2s_obj.evaluate(y_test, pred)
        mlflow.log_metric("r2 score", r2s)

        rmse_obj = RMSE()
        rmse = rmse_obj.evaluate(y_test, pred)
        mlflow.log_metric("rmse", rmse)

        logging.info(f'MSE: {mse}, R2Score: {r2s}, RMSE: {rmse}')

        return r2s, rmse
    except Exception as e:
        logging.error(f'Error in evaluation: {e}')
        raise e
