# Phishing-Email-Detection-Using-Natural-Language-Processing-NLP-
#focused on threat intelligence aggregation and phishing email detection using NLP (Natural Language Processing). This project will help in identifying phishing attempts in email content by analyzing the text and extracting patterns indicative of phishing attacks.
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords

# Download stopwords for text preprocessing
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Preprocess the email content (removing special characters, stopwords, etc.)
def preprocess_email(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load the phishing email dataset (example dataset can be CICIDS 2018 or any other dataset)
# For the sake of the example, let's create a simple dataset
data = {
    'email_content': [
        "Congratulations! You've won a $1000 gift card. Claim now!",
        "Your account has been compromised. Please reset your password immediately.",
        "Bank statement for the month is available for download.",
        "Click here to confirm your PayPal payment.",
        "Hi, just wanted to check in to see how you are doing.",
        "Important update on your account. Action required!"
    ],
    'label': ['Phishing', 'Phishing', 'Legitimate', 'Phishing', 'Legitimate', 'Phishing']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocess the email content
df['processed_email'] = df['email_content'].apply(preprocess_email)

# Split the dataset into features (X) and labels (y)
X = df['processed_email']
y = df['label'].map({'Phishing': 1, 'Legitimate': 0})  # Phishing -> 1, Legitimate -> 0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical features using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Create a machine learning pipeline with vectorizer and classifier
model = make_pipeline(tfidf_vectorizer, RandomForestClassifier(n_estimators=100, random_state=42))

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to detect phishing in real-time email content
def detect_phishing(email_content):
    processed_email = preprocess_email(email_content)
    prediction = model.predict([processed_email])
    if prediction == 1:
        print("ALERT! Phishing email detected!")
    else:
        print("Legitimate email.")

# Simulate real-time phishing detection
new_email = "Dear user, your account has been compromised. Click here to secure your account."
detect_phishing(new_email)
