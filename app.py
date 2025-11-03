from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
load_dotenv()  # loads .env file
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_alert_email(to_email, subject, body):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Error sending email: {e}")


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
    emergency_email = request.form['emergency_email']

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = labels[torch.argmax(scores).item()]
        confidence = torch.max(scores).item() * 100

    data = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Journal': text,
        'Sentiment': predicted_label,
        'Confidence (%)': round(confidence, 2),
        'Emergency Email': emergency_email
    }
    df = pd.DataFrame([data])

    if not os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, index=False)
    else:
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)

    # Send email if mood is negative
    if predicted_label == "Negative" and emergency_email:
        send_alert_email(
            to_email=emergency_email,
            subject="Mood Alert: Negative Mood Detected",
            body=f"A negative mood was detected in the journal entry:\n\n\"{text}\"\n\nConfidence: {round(confidence, 2)}%"
        )

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
