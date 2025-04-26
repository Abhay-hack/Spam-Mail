import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack
from flask import Flask, request, render_template

app = Flask(__name__)

# Custom Transformer for Text Preprocessing
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._preprocess(text) for text in X]

    def _preprocess(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'(https?://\S+|www\.\S+)', ' URL ', text)
        text = text.translate(str.maketrans('', '', string.punctuation.replace('/', '')))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

# Custom Transformer for Feature Engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_list = [self._extract_features(text) for text in X]
        return pd.DataFrame(features_list).astype(float).values

    def _extract_features(self, text):
        if not isinstance(text, str):
            text = str(text)
        return {
            'text_length': float(len(text)),
            'has_capitalized_word': float(bool(re.search(r'[A-Z]', text))),
        }

# Load the trained model components
try:
    text_preprocessor = joblib.load('text_preprocessor.joblib')
    feature_engineer = joblib.load('feature_engineer.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    classifier = joblib.load('spam_classifier.joblib')
    print("Model components loaded successfully for web app.")
except FileNotFoundError:
    print("Error: One or more model components not found. Please run the script to train and save the model first.")
    exit(1)
except Exception as e:
    print(f"Error loading model components for web app: {e}")
    exit(1)

# Predict new SMS (using loaded components)
def predict_sms(text):
    try:
        processed_text = text_preprocessor.transform([text])
        text_tfidf = tfidf_vectorizer.transform(processed_text)
        text_features = feature_engineer.transform([text])
        combined_features = hstack([text_tfidf, text_features]).toarray()
        prediction = classifier.predict(combined_features)[0]
        probability = classifier.predict_proba(combined_features)[0][prediction]
        return "Spam" if prediction == 1 else "Ham", f"{probability:.4f}"
    except Exception as e:
        print(f"Error during prediction for web app: {e}")
        return "Error", "0.0000"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        sms_text = request.form['sms_text']
        if sms_text:
            prediction, probability = predict_sms(sms_text)
    return render_template('index.html', prediction=prediction, probability=probability)

# Visualize confusion matrix (keep this function)
def plot_confusion_matrix(cm, filename='confusion_matrix_advanced.png'):
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

# Download NLTK resources (keep this function)
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)  # For lemmatization
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        exit(1)

# Load the dataset (keep this function)
def load_data(file_path='spam.csv'):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found in the current directory.")
        exit(1)
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

# Custom Transformer for Text Preprocessing (keep this class)
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._preprocess(text) for text in X]

    def _preprocess(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'(https?://\S+|www\.\S+)', ' URL ', text)
        text = text.translate(str.maketrans('', '', string.punctuation.replace('/', '')))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

# Custom Transformer for Feature Engineering (keep this class)
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_list = [self._extract_features(text) for text in X]
        return pd.DataFrame(features_list).astype(float).values

    def _extract_features(self, text):
        if not isinstance(text, str):
            text = str(text)
        return {
            'text_length': float(len(text)),
            'has_capitalized_word': float(bool(re.search(r'[A-Z]', text))),
        }

# Train and evaluate model (modify this to only save components)
def train_model(df, model_filename='spam_model.joblib', vectorizer_filename='tfidf_vectorizer.joblib'):
    try:
        X = df['text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Text Preprocessor
        text_preprocessor = TextPreprocessor()
        X_train_processed_text = text_preprocessor.fit_transform(X_train)
        X_test_processed_text = text_preprocessor.transform(X_test)

        # Feature Engineer
        feature_engineer = FeatureEngineer()
        X_train_features = feature_engineer.fit_transform(X_train)
        X_test_features = feature_engineer.transform(X_test)

        # TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_processed_text)
        X_test_tfidf = tfidf_vectorizer.transform(X_test_processed_text)

        # Concatenate the TF-IDF features with the engineered features
        X_train_combined = hstack([X_train_tfidf, X_train_features]).toarray()
        X_test_combined = hstack([X_test_tfidf, X_test_features]).toarray()

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)

        # Train Logistic Regression
        classifier = LogisticRegression(C=10, class_weight='balanced', max_iter=1000, random_state=42)
        classifier.fit(X_train_resampled, y_train_resampled)
        y_pred = classifier.predict(X_test_combined)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Save the individual components
        joblib.dump(text_preprocessor, 'text_preprocessor.joblib')
        joblib.dump(feature_engineer, 'feature_engineer.joblib')
        joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
        joblib.dump(classifier, 'spam_classifier.joblib')
        print("Trained model components saved.")

        return text_preprocessor, feature_engineer, tfidf_vectorizer, classifier, accuracy, precision, recall, f1, cm, report, X_test, y_test, y_pred
    except Exception as e:
        print(f"Error training model: {e}")
        exit(1)

def main():
    download_nltk_resources()
    df = load_data('spam.csv')
    print("Dataset loaded.")
    # Train the model and get the components
    text_preprocessor, feature_engineer, tfidf_vectorizer, classifier, accuracy, precision, recall, f1, cm, report, X_test, y_test, y_pred = train_model(df)

    print("\nModel Evaluation (Logistic Regression with Feature Engineering):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:\n", report)

    plot_confusion_matrix(cm, filename='confusion_matrix_advanced.png')
    print("Confusion matrix saved as 'confusion_matrix_advanced.png'")

    # The web app will run when this script is executed directly
    app.run(debug=True)

if __name__ == "__main__":
    main()