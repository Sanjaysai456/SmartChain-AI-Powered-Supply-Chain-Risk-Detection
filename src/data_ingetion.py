import os
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.join("artifacts")
    raw_data_path: str = os.path.join(artifacts_dir,  "raw_data.csv")
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")

class DataIngestion:
    def __init__(self, source_path: str = "C:/ML_Projects/Risk_Management/notebooks/data.csv"):
        self.source_path = source_path
        self.config = DataIngestionConfig()

    def run(self):
        print("Starting DataIngestion")
        try:
            df = pd.read_csv(self.source_path)
            print(f"Dataloaded from: {self.source_path}")
            

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            print(f"Raw data saved to: {self.config.raw_data_path}")

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            print("TrainTest split completed ")
            print(f"Training data: {train_df.shape}, Testing data: {test_df.shape}")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            print(f"Error during data ingestion: {e}")
            raise e

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.run()
    print(f"\nData Ingestion Completed.\nTrain Path: {train_path}\nTest Path: {test_path}")
