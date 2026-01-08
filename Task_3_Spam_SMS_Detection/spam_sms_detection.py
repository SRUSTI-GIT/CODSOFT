# Task 4: Spam SMS Detection
# TF-IDF + Naive Bayes (Baseline ML Model)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

# Encode labels
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# Model Training
# -----------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
