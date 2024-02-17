import logging

import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_loc: str):
        self.data_loc = data_loc

    def get_data(self):
        logging.info(f"Getting data from {self.data_loc}")
        return pd.read_csv(self.data_loc)

@step
def ingest_df(
    data_loc: str
) -> pd.DataFrame:
    ingest_data = IngestData(data_loc)
    df = ingest_data.get_data()
    logging.info(f"Data Frame ingestered: {df.info}")
    return df