import logging

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

import numpy as np
import pandas as pd
import json

from steps.ingest_data import ingest_df
from steps.data_preprocess import preprocess_df
from steps.model_train import model_trainer
from steps.model_evaluate import evalualte_model
from src.data_preprocessor import DataColPreprocessorStrategy, DataPrep

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeployTriggerConfig(BaseParameters):
    """
    Configurations for the model
    """
    min_accuracy: float = 0.50

@step
def deploy_trigger(
    accuracy: float,
    config: DeployTriggerConfig
) -> bool:
    if accuracy < config.min_accuracy:
        return False
    return True

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def cd_pipeline(
    data_loc: str,
    min_accuracy: float = 0.5,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_loc=data_loc)
    X_train, X_test, y_train, y_test = preprocess_df(df)
    model = model_trainer(X_train, X_test, y_train, y_test)
    r2s, rmse = evalualte_model(model, X_test, y_test)
    deploy_ruling = deploy_trigger(r2s)

    mlflow_model_deployer = mlflow_model_deployer_step(
        model = model,
        deploy_decision=deploy_ruling,
        workers = workers,
        timeout = timeout
    )

def get_data_for_test():
    try: 
        df = pd.read_csv("/Users/ajaybalasa/Documents/workspace/zenml/mlops-projects/zenml/customer-satisf/data/olist_customers_dataset.csv")
        df = df.sample(n=100) # Get 100 samples
        col_preprocessor = DataColPreprocessorStrategy()
        df_cleaned = DataPrep(df, col_preprocessor)
        df = df_cleaned.handle_data()
        df .drop(columns=["review_score"], axis=1, inplace=True)
        df_json = df.to_json(orient="split")
        return df_json
    except Exception as e:
        logging.error(f'Error in creating data for prediction: {e}')
        raise e

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data

@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Loads the prediction service by the deploy pipeline."""
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing runs with same pipeline name, step name and model name
    existing_runs = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_runs:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_runs)
    print(type(existing_runs))
    return existing_runs[0]

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)