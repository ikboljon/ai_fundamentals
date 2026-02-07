# model/train_model.py
"""
Train simple TF-IDF + LogisticRegression classifier on sklearn 20newsgroups.
Saves: classifier.joblib containing {'vect':vectorizer, 'clf':classifier, 'labels':labels}
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import argparse

def train_and_save(model_path="model/classifier.joblib"):
    print("Loading 20 newsgroups dataset...")
    data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
    X = data.data
    y = data.target
    labels = data.target_names
    print(f"Samples: {len(X)}  Classes: {len(labels)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Fitting TF-IDF vectorizer...")
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')

    print("Training classifier...")
    Xtr = vect.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, y_train)

    print("Evaluating on test set...")
    Xte = vect.transform(X_test)
    ypred = clf.predict(Xte)
    print(classification_report(y_test, ypred, target_names=labels))

    print(f"Saving model to {model_path} ...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'vect': vect, 'clf': clf, 'labels': labels}, model_path)
    print("Saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="model/classifier.joblib")
    args = parser.parse_args()
    train_and_save(args.out)
