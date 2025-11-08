from flask import Flask, render_template, request
import pandas as pd
from src.predict import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = CustomData(
            temperature=float(request.form['temperature']),
            humidity=float(request.form['humidity']),
            vibration_level=float(request.form['vibration_level']),
            stock_quantity=int(request.form['stock_quantity']),
            supplier_rating=float(request.form['supplier_rating']),
            delivery_delay_days=int(request.form['delivery_delay_days']),
            location=request.form['location'],
            inventory_status=request.form['inventory_status'],
            logistics_partner=request.form['logistics_partner'],
            shipment_status=request.form['shipment_status'],
            weather_condition=request.form['weather_condition'],
            social_media_feed=request.form['social_media_feed'],
            news_alert=request.form['news_alert'],
            system_log_message=request.form['system_log_message']
        )

       
        df_input = data.get_data_as_df()

       
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df_input)

        return render_template('index.html', prediction_text=f"Predicted Supply Chain Risk: {prediction}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
