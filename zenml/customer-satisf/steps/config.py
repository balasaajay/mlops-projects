from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """
    Configurations for the model
    """
    model_name: str = "LinearRegression"

