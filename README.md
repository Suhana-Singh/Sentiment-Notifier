🔹 Overview

Happy Harbour is a simple and elegant Flask web app powered by RoBERTa, that analyzes your mood from journal entries.
It helps you reflect on your emotional well-being while storing your past entries for review and trend visualization.
If your mood shows negative trend, it will notify to your family member or you can opt for counselling.

⚙️ Features

* Uses RoBERTa for sentiment classification (Positive / Neutral / Negative)
* Lets you write and analyze daily journals
* Saves your entries with timestamps in a CSV file
* Displays recent mood history and charts
* Beginner-friendly interface with HTML/CSS frontend
  
🧩 Tech Stack

 Component | Technology  
 
 Backend    Flask                                                 
 NLP Model  RoBERTa (cardiffnlp/twitter-roberta-base-sentiment) 
 Frontend   HTML, CSS                                             
 Storage    CSV file                                              
 Optional   JavaScript for chart visualization                    

🗂 Folder Structure

Happy-Harbour/
│
├── app.py
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   └── chart.html
├── static/
│   └── style.css
├── journals.csv
└── requirements.txt

🧭 Future Add-ons

📱 Notifications for negative moods over time
📈 Weekly sentiment trend charts
💾 Database integration (SQLite / Firebase)
🧍 User authentication for personal dashboards

OUTPUT
<img width="1435" height="803" alt="Screenshot 2025-11-11 at 23 21 13" src="https://github.com/user-attachments/assets/2b55e792-4b9b-407e-af5d-892c7d84f9f5" />


