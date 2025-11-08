import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import save_object


@dataclass
class TransformConfig:
    train_arr_path: str = os.path.join("artifacts", "data_transformation", "train_arr.npy")
    test_arr_path: str = os.path.join("artifacts", "data_transformation", "test_arr.npy")
    encoder_path: str = os.path.join("artifacts", "data_transformation", "label_encoders.pkl")

class DataTransform:
    def __init__(self):
        self.cfg = TransformConfig()

    def start_transformation(self, file_path):
        try:
            df = pd.read_csv(file_path)
            print("Data loaded for transformation")

            
            df['expected_delivery_date'] = pd.to_datetime(df['expected_delivery_date'])
            df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'])
            df['delivery_delay_days'] = (df['actual_delivery_date'] - df['expected_delivery_date']).dt.days

            
            drop_cols = [
                'timestamp', 'device_id', 'order_id', 'order_placed_date', 'supplier_id',
                'expected_delivery_date', 'actual_delivery_date'
            ]
            df.drop(drop_cols, axis=1, inplace=True)
            print("dropeed unwanted columns")

            
            cat_cols = ['location', 'inventory_status', 'logistics_partner', 'shipment_status', 'weather_condition']
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

           
            bool_cols = df.select_dtypes(include='bool').columns
            
            df[bool_cols] = df[bool_cols].astype(int)

          
            encode_cols = ['social_media_feed', 'news_alert', 'system_log_message']
            encoders = {}
            for col in encode_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            print("Label encoding completed")

           
            X = df.drop('manual_risk_label', axis=1)
            y = df['manual_risk_label']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

           
            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

           
            os.makedirs(os.path.dirname(self.cfg.train_arr_path), exist_ok=True)
            np.save(self.cfg.train_arr_path, train_arr)
            np.save(self.cfg.test_arr_path, test_arr)
            save_object(self.cfg.encoder_path, encoders)
            np.save(os.path.join("artifacts", "data_transformation", "feature_names.npy"), X.columns.to_numpy())


            print("Transformation complete arrays and encoders saved.")
            return self.cfg.train_arr_path, self.cfg.test_arr_path, self.cfg.encoder_path

        except Exception as e:
            print(f"Error during transformation: {e}")
            raise e
if __name__ == "__main__":
    
    transformer = DataTransform()
    train_arr, test_arr, encoder_path = transformer.start_transformation("artifacts/raw_data.csv")

    print("\n Data Transformation completed ")
    print(f"Train array saved at: {train_arr}")
    print(f"Test array saved at: {test_arr}")
    print(f"Encoders saved at: {encoder_path}")
