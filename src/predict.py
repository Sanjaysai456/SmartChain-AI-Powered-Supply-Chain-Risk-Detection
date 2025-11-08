import os
import pandas as pd
import numpy as np
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        
        self.model_path = os.path.join("artifacts", "model_trainer", "supply_chain_risk_model.pkl")
        self.encoder_path = os.path.join("artifacts", "data_transformation", "label_encoders.pkl")

       

        self.model = load_object(self.model_path)
        self.encoders = load_object(self.encoder_path)

       
        self.feature_names_path = os.path.join("artifacts", "data_transformation", "feature_names.npy")
        
        self.feature_names = np.load(self.feature_names_path, allow_pickle=True).tolist()
        

    def predict(self, input_df: pd.DataFrame):
        
        try:
           
            for col in ['social_media_feed', 'news_alert', 'system_log_message']:
                if col in input_df.columns:
                    encoder = self.encoders.get(col)
                    if encoder:
                        input_df[col] = encoder.transform(input_df[col])
                    else:
                        input_df[col] = 0

           
            cat_cols = ['location', 'inventory_status', 'logistics_partner', 'shipment_status', 'weather_condition']
            input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

           
            if self.feature_names:
                for col in self.feature_names:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[self.feature_names]
            
           
            input_array = input_df.to_numpy()

         
            prediction = self.model.predict(input_array)[0]
            labels = {0: "No Risk", 1: "Moderate Risk", 2: "High Risk"}
            return labels.get(prediction, "Unknown")

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise e


class CustomData:


    def __init__(
        self,
        temperature: float,
        humidity: float,
        vibration_level: float,
        stock_quantity: int,
        supplier_rating: float,
        delivery_delay_days: int,
        location: str,
        inventory_status: str,
        logistics_partner: str,
        shipment_status: str,
        weather_condition: str,
        social_media_feed: str,
        news_alert: str,
        system_log_message: str
    ):
        self.temperature = temperature
        self.humidity = humidity
        self.vibration_level = vibration_level
        self.stock_quantity = stock_quantity
        self.supplier_rating = supplier_rating
        self.delivery_delay_days = delivery_delay_days
        self.location = location
        self.inventory_status = inventory_status
        self.logistics_partner = logistics_partner
        self.shipment_status = shipment_status
        self.weather_condition = weather_condition
        self.social_media_feed = social_media_feed
        self.news_alert = news_alert
        self.system_log_message = system_log_message

    def get_data_as_df(self):
        
        data = {
            "temperature": [self.temperature],
            "humidity": [self.humidity],
            "vibration_level": [self.vibration_level],
            "stock_quantity": [self.stock_quantity],
            "supplier_rating": [self.supplier_rating],
            "delivery_delay_days": [self.delivery_delay_days],
            "location": [self.location],
            "inventory_status": [self.inventory_status],
            "logistics_partner": [self.logistics_partner],
            "shipment_status": [self.shipment_status],
            "weather_condition": [self.weather_condition],
            "social_media_feed": [self.social_media_feed],
            "news_alert": [self.news_alert],
            "system_log_message": [self.system_log_message],
        }
        return pd.DataFrame(data)







