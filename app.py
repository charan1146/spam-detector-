import streamlit as st
import pandas as pd

# Machine learning imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page settings
st.set_page_config(
    page_title="AI Spam Email Detector",
    page_icon="📧"
)

# Title
st.title("📧 AI Spam Email Detector")

# Description
st.write("Detect whether an email/message is Spam or Not Spam.")

# Load dataset
data = pd.read_csv("emails.csv")

# Convert text into numbers
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data["text"])

# Labels
y = data["label"]

# Train model
model = MultinomialNB()

model.fit(X, y)

# User input
message = st.text_area("Enter your message:")

# Prediction
if message:

    # Convert input into numbers
    message_vector = vectorizer.transform([message])

    # Predict
    prediction = model.predict(message_vector)

    # Output
    if prediction[0] == "spam":
        st.error("🚨 This message is SPAM")

    else:
        st.success("✅ This message is NOT SPAM")

# Footer
st.write("---")
st.caption("Developed using Python, Streamlit, and Machine Learning")