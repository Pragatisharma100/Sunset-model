from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("model/lstm_model.h5", compile=False)
scaler = joblib.load("model/scaler.save")

# Load sunspot data
df = pd.read_csv("SN_m_tot_V2.0.csv", sep=';', header=None)
df.columns = [
    "Year", "Month", "DecimalDate",
    "SunspotNumber", "StandardDeviation",
    "Observations", "Definitive"
]
df = df[df['SunspotNumber'] != -1]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    labels = []
    values = []
    year = None

    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            filtered_df = df[df['Year'] >= year]

            # ✅ Get the FIRST 12 rows after the input year
            history_12_df = filtered_df.head(12)

            if len(history_12_df) < 12:
                prediction = "Insufficient data after selected year."
            else:
                sunspot_values = history_12_df["SunspotNumber"].values.reshape(-1, 1)
                scaled_values = scaler.transform(sunspot_values)
                input_seq = scaled_values.reshape(1, 12, 1)
                pred = model.predict(input_seq)
                predicted_value = scaler.inverse_transform(pred)[0][0]
                prediction = round(predicted_value, 2)

                # For chart
                labels = history_12_df.apply(lambda row: f"{int(row['Month'])}/{int(row['Year'])}", axis=1).tolist()
                values = history_12_df["SunspotNumber"].tolist()

        except Exception as e:
            prediction = "Error in processing: " + str(e)

    return render_template('index.html',
                           prediction=prediction,
                           labels=labels,
                           values=values,
                           year=year)
