import logging

import pandas as pd
from zenml import step 

from src.data_preprocessor import DataPrep, DataColPreprocessorStrategy, DataSplitStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def preprocess_df(
    df: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"], 
    Annotated[pd.DataFrame, "X_test"], 
    Annotated[pd.Series, "y_train"], 
    Annotated[pd.Series, "y_test"], 
]:
    """
    Preprocess the dataset and split for train/test the model.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        X_train: Train dataset
        X_test: Test dataset
        y_train: Train target/label
        y_test: Test target/label
    """
    try:
        logging.info(f'Preprocessing started...')

        col_process_strategy = DataColPreprocessorStrategy()
        df = DataPrep(df, col_process_strategy).handle_data()
        
        split_strategy = DataSplitStrategy()
        X_train, X_test, y_train, y_test = DataPrep(df, split_strategy).handle_data()

        logging.info(f'Preprocessing done!')
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f'Error in data cleaning: {e}')
        raise e
