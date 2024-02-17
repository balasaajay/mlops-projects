import logging

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from typing import Union
from sklearn.model_selection import train_test_split


"""
uses Strategy Design Pattern for flexible/readable and reduce repetition
"""

class DataPreprocessorStrategy(ABC):

    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataColPreprocessorStrategy(DataPreprocessorStrategy):
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
                "customer_zip_code_prefix", 
                "order_item_id"
            ], axis=1) # Drop unnecessary columns
            
            df['product_weight_g'] = df['product_weight_g'].fillna(df['product_weight_g'].median(), inplace=True)
            df['product_length_cm'] = df['product_length_cm'].fillna(df['product_length_cm'].median(), inplace=True)
            df['product_width_cm'] = df['product_width_cm'].fillna(df['product_width_cm'].median(), inplace=True)
            df['product_height_cm'] = df['product_height_cm'].fillna(df['product_height_cm'].median(), inplace=True)
            df['review_comment_message'] = df['review_comment_message'].fillna("Review Not Available", inplace=True)

            # Drop categorical columns
            df = df.select_dtypes(include=[np.number])

            return df
        except Exception as e:
            logging.error(f'Error in data preprocessing: {e}')
            raise e

class DataSplitStrategy(DataPreprocessorStrategy):
    """
    Split data for train and test
    """
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = df.drop(["review_score"], axis=1)
            y = df["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=400)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f'Error in data splitting: {e}')
            raise e

class DataPrep:
    """
        Preprocess and split the dataset for train and test
    """
    def __init__(self, df: pd.DataFrame, strategy: DataPreprocessorStrategy):
        self.df = df
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.error(f'Error in data preprocessing: {e}')
            raise e
