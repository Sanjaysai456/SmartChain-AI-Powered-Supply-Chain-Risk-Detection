import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_object

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model_trainer", "supply_chain_risk_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.cfg = ModelTrainerConfig()

    def start_training(self, train_arr_path, test_arr_path):
        try:
            
            train_arr = np.load(train_arr_path, allow_pickle=True)
            test_arr = np.load(test_arr_path, allow_pickle=True)

            print("Data loaded successfully  training")

           
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

           
            rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            rf.fit(X_train, y_train)
            print("Random Forest  trained successfully")

            y_pred = rf.predict(X_test)

           
            acc = accuracy_score(y_test, y_pred)
            print(f"\nAccuracy: {acc}")
            print("\nClassification Report:\n", classification_report(y_test, y_pred))

            
            os.makedirs(os.path.dirname(self.cfg.model_path), exist_ok=True)
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title("Random Forest Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            plt.savefig(os.path.join("artifacts", "model_trainer", "confusion_matrix.png"))
            plt.close()

           
            save_object(self.cfg.model_path, rf)
            print(f"model saved at: {self.cfg.model_path}")

            return {"accuracy": acc, "model_path": self.cfg.model_path}

        except Exception as e:
            print(f"Error during model training: {e}")
            raise e


if __name__ == "__main__":
   
    train_arr_path = os.path.join("artifacts", "data_transformation", "train_arr.npy")
    test_arr_path = os.path.join("artifacts", "data_transformation", "test_arr.npy")

    if not (os.path.exists(train_arr_path) and os.path.exists(test_arr_path)):
        print("Transformed data not found. Please Run datatransformation first")
    else:
        trainer = ModelTrainer()
        result = trainer.start_training(train_arr_path, test_arr_path)

        print("\nModel Training completed successfully!")
        print(f"Model Accuracy: {result['accuracy']}")
        print(f"Model saved at: {result['model_path']}")
