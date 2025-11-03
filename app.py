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

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load RoBERTa model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = ['Negative', 'Neutral', 'Positive']
CSV_FILE = "journals.csv"


# ---------- EMAIL ALERT FUNCTION ----------
def send_alert_email(to_email, user_text, sentiment, confidence):
    sender_email = os.getenv("EMAIL_USER")
    sender_pass = os.getenv("EMAIL_PASS")

    subject = "🚨 Mood Alert: Negative Sentiment Detected"
    body = f"""
    Hello,

    The system detected a NEGATIVE mood from your contact's journal entry.

    Journal Entry:
    "{user_text}"

    Sentiment: {sentiment}
    Confidence: {confidence}%

    Please check in with them.

    Regards,
    Happy Harbour 💖
    """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_pass)
            server.send_message(msg)
            print(f"Alert email sent to {to_email}")
    except Exception as e:
        print(f"Error sending email: {e}")


# ---------- ROUTINE SUGGESTION FUNCTION ----------
def generate_routine():
    return [
        "🧘 Take 10 minutes to meditate or breathe deeply.",
        "🚶 Go for a short walk outdoors and get some sunlight.",
        "📓 Write down 3 positive things about yourself.",
        "💧 Drink water and stretch a bit.",
        "🎶 Listen to your favorite uplifting song.",
        "☎️ Talk to a friend or loved one.",
        "🌅 Try sleeping early and rest well tonight."
    ]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['journal']
    emergency_email = request.form.get('emergency_email', '')

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

    # Handle negative sentiment
    routine = []
    if predicted_label == "Negative":
        routine = generate_routine()
        if emergency_email:
            send_alert_email(emergency_email, text, predicted_label, round(confidence, 2))

    return render_template(
        'result.html',
        text=text,
        sentiment=predicted_label,
        confidence=round(confidence, 2),
        routine=routine
    )





@app.route('/history')
def history():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, usecols=[0, 1, 2, 3], on_bad_lines='skip')
        last_entries = df.tail(5).iloc[::-1]
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
        df = df.tail(10)
        data = {
            "labels": df["Timestamp"].tolist(),
            "sentiments": df["Sentiment"].tolist()
        }
        return data
    else:
        return {"labels": [], "sentiments": []}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
