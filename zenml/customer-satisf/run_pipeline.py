from pipelines.end2end_pipeline import end2end_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    end2end_pipeline(data_loc="/Users/ajaybalasa/Documents/workspace/zenml/mlops-projects/zenml/customer-satisf/data/olist_customers_dataset.csv")
