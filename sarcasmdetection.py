import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import nltk
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load datasets
train_df = pd.read_csv('C:/Users/Nisa/Desktop/Sarcasm Detection/train-balanced-sarc.csv')

# Explore dataset
print("Dataset Head:")
print(train_df.head())

# Check for missing values
print("Missing values:")
print(train_df.isnull().sum())

# Assuming the columns 'text' contains the text and 'label' contains the target
train_df = train_df[['text', 'label']].dropna()

# Preprocessing functions
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back to string
    return ' '.join(tokens)

# Apply preprocessing
train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)

# Split dataset
X = train_df['cleaned_text']
y = train_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training with Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optional: Save the model and vectorizer
import joblib
joblib.dump(model, 'sarcasm_detection_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')