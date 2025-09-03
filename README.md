# AI Pipeline Engine

A modular, test-driven AI pipeline built using Python best practices and OOP design patterns.

## Features
- Text normalization (lowercase, punctuation removal, stopwords)
- Pluggable processing pipeline (Strategy pattern)
- Feature extraction (Bag-of-Words)
- Sentiment classifier with scikit-learn
- Evaluation step (accuracy, precision, recall, F1)
- 100% tested with Pytest

## Usage

```bash
pytest tests/
```

## Structure
```bash
ai-pipeline-engine/
├── text_pipeline/                 # Main Python package
│   ├── __init__.py
│   ├── text_pipeline.py
│   └── steps/
│       ├── __init__.py
│       ├── base.py
│       ├── lowercase.py
│       ├── remove_punctuation.py
│       ├── remove_stopwords.py
│       ├── bag_of_words.py
│       ├── text_classifier.py
│       └── evaluation.py
├── tests/
│   └── test_text_pipeline.py
├── train_classifier.py           # Script for training model
├── sentiment_model.pkl           # Trained model (can ignore in .gitignore)
├── requirements.txt
├── README.md
├── .gitignore
```
