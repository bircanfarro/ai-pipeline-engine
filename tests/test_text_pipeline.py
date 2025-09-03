
from text_pipeline import TextPipeline
from steps.lowercase import Lowercase
from steps.remove_punctuation import RemovePunctuation
from steps.remove_stopwords import RemoveStopwords
from steps.bag_of_words import BagOfWordVectorizer
from steps.text_classifier import TextClassifier

def test_pipeline_applies_all_steps():
    pipeline = TextPipeline([
        Lowercase(),
        RemovePunctuation(),
        RemoveStopwords(["the", "is", "and"]),
        BagOfWordVectorizer()
    ])
    result = pipeline.process("The QUICK, brown fox... jumps! And is gone.")
    assert result == {
        'quick':1, 'brown':1, 'fox':1, 'jumps':1, 'gone':1
    }

def test_sentiment_prediction():
    pipeline = TextPipeline([
        TextClassifier()  # assumes model is already trained
    ])
    assert pipeline.process("I love it") == "positive"
    assert pipeline.process("This is terrible") == "negative"
