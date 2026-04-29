import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os

def train_security_model(data_path="data/messages.csv", model_path="models/phishing_pipeline.pkl"):
    """
    Trains a TF-IDF + Naive Bayes pipeline to detect phishing messages.
    """
    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found.")
        return None
    
    df = pd.read_csv(data_path)
    
    X = df['text']
    y = df['label'] # 'safe' or 'phishing'
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', MultinomialNB())
    ])
    
    pipeline.fit(X, y)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    
    return pipeline

def load_security_model(model_path="models/phishing_pipeline.pkl"):
    """Loads a pre-trained security model."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def scan_message(pipeline, message_text):
    """
    Takes a single string, returns classification and risk score.
    Risk score is the probability of the message being 'phishing'.
    """
    if not pipeline:
        return "Unknown", 0
        
    prediction = pipeline.predict([message_text])[0]
    probabilities = pipeline.predict_proba([message_text])[0]
    
    # Assuming the classes are ['phishing', 'safe'] alphabetically.
    # We should get the index for 'phishing'.
    classes = pipeline.classes_
    phishing_idx = list(classes).index('phishing')
    
    risk_score = round(probabilities[phishing_idx] * 100, 2)
    
    return prediction, risk_score
