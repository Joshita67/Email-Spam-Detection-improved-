import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd

# Load components
model = joblib.load("improved_spam_model.pkl")
vectorizer = joblib.load("improved_vectorizer.pkl")
feature_names = joblib.load("engineered_feature_names.pkl")

# Custom feature extractor
def extract_features(text):
    features = {
        'has_urgent_words': int(bool(re.search(r"immediate|urgent|action required|24 hours", text, re.IGNORECASE))),
        'has_links': int("http" in text or "www" in text or "bit.ly" in text),
        'has_attachment_terms': int(bool(re.search(r"\\.zip|\\.pdf|attachment", text, re.IGNORECASE))),
        'phishing_keywords': int(bool(re.search(r"invoice|bank|login|payment|verify|credentials", text, re.IGNORECASE))),
        'capital_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),
        'has_html_tags': int(bool(re.search(r"<.*?>", text))),
        'excessive_punctuation': int(text.count('!') > 1 or text.count('.') > 3),
        'text_len': len(text),
    }
    return pd.DataFrame([features])

# Streamlit UI
st.title("📧 Spam Email Classifier (Improved)")
input_text = st.text_area("Paste the email message here:")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some email text.")
    else:
        custom_features = extract_features(input_text)
        tfidf_vector = vectorizer.transform([input_text])
        from scipy.sparse import hstack
        final_input = hstack([tfidf_vector, custom_features])
        
        pred = model.predict(final_input)[0]
        pred_proba = model.predict_proba(final_input)[0][1]

        label = "🚨 Spam" if pred == 1 else "✅ Ham"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {pred_proba:.2%}")
