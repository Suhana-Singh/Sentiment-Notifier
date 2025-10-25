from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Load RoBERTa model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = ['Negative', 'Neutral', 'Positive']

# CSV file to store user entries
CSV_FILE = "journals.csv"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['journal']

    # Model prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = labels[torch.argmax(scores).item()]
        confidence = torch.max(scores).item() * 100

    # Save entry to CSV
    data = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Journal': text,
        'Sentiment': predicted_label,
        'Confidence (%)': round(confidence, 2)
    }

    df = pd.DataFrame([data])

    if not os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, index=False)
    else:
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)

    return render_template('result.html',
                           text=text,
                           sentiment=predicted_label,
                           confidence=round(confidence, 2))




@app.route('/history')
def history():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        last_entries = df.tail(5).iloc[::-1]  # show last 5, newest first
        entries = last_entries.to_dict(orient='records')
    else:
        entries = []
    return render_template('history.html', entries=entries)

@app.route('/chart')
def chart():
    return render_template('chart.html')


@app.route('/chart-data')
def chart_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        # Use only the last 10 entries
        df = df.tail(10)
        data = {
            "labels": df["Timestamp"].tolist(),
            "sentiments": df["Sentiment"].tolist()
        }
        return data
    else:
        return {"labels": [], "sentiments": []}
    

if __name__ == "__main__":
    app.run(debug=True)
