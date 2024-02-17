from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.data_preprocess import preprocess_df
from steps.model_train import model_trainer
from steps.model_evaluate import evalualte_model

@pipeline(enable_cache=False)
def end2end_pipeline(data_loc: str):
    df = ingest_df(data_loc)
    X_train, X_test, y_train, y_test = preprocess_df(df)
    model = model_trainer(X_train, X_test, y_train, y_test)
    r2, rmse = evalualte_model(model, X_test, y_test)