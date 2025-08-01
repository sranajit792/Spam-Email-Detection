
# spam_detector_fuzzy.py

import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['message'] = df['message'].apply(clean_text).apply(preprocess)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Vectorize
cv = CountVectorizer()
X = cv.fit_transform(df['message']).toarray()
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- Fuzzy Logic Integration ---

# Define fuzzy variables
spam_prob = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'spam_prob')
risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

spam_prob['low'] = fuzz.trimf(spam_prob.universe, [0, 0, 0.4])
spam_prob['medium'] = fuzz.trimf(spam_prob.universe, [0.2, 0.5, 0.8])
spam_prob['high'] = fuzz.trimf(spam_prob.universe, [0.6, 1, 1])

risk['low'] = fuzz.trimf(risk.universe, [0, 0, 40])
risk['medium'] = fuzz.trimf(risk.universe, [30, 50, 70])
risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

# Define rules
rule1 = ctrl.Rule(spam_prob['low'], risk['low'])
rule2 = ctrl.Rule(spam_prob['medium'], risk['medium'])
rule3 = ctrl.Rule(spam_prob['high'], risk['high'])

risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
risk_eval = ctrl.ControlSystemSimulation(risk_ctrl)

# Prediction with fuzzy logic
def predict_custom_fuzzy(msg):
    clean = preprocess(clean_text(msg))
    vect = cv.transform([clean]).toarray()

    prob_spam = model.predict_proba(vect)[0][1]  # Probability of being spam

    # Fuzzy logic processing
    risk_eval.input['spam_prob'] = prob_spam
    risk_eval.compute()
    fuzzy_risk = risk_eval.output['risk']

    print(f"\nMessage: {msg}")
    print(f"Spam Probability: {prob_spam:.2f}")
    print(f"Fuzzy Risk Level: {fuzzy_risk:.2f}")

    if fuzzy_risk < 40:
        return "✅ Not Spam"
    elif fuzzy_risk < 70:
        return "⚠️ Suspicious"
    else:
        return "❌ Spam"

# Test message
test_msg = "Free entry in 2 a weekly competition to win tickets!"
result = predict_custom_fuzzy(test_msg)
print("Final Verdict:", result)
