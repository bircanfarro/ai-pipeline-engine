# train_classifier.py

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

X = [
    "I love this product",
    "This is the worst thing",
    "Absolutely fantastic",
    "I hate it"
]
y = ["positive", "negative", "positive", "negative"]

pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

pipeline.fit(X, y)

joblib.dump(pipeline, "sentiment_model.pkl")
