import pytest
from text_pipeline.text_pipeline import TextPipeline
from text_pipeline.steps.lowercase import Lowercase
from text_pipeline.steps.remove_punctuation import RemovePunctuation
from text_pipeline.steps.remove_stopwords import RemoveStopwords
from text_pipeline.steps.bag_of_words import BagOfWordVectorizer
from text_pipeline.steps.text_classifier import TextClassifier
from text_pipeline.steps.evaluation import EvaluationStep

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

def test_evaluation_step_metrics():
    true = ["positive", "negative", "positive"]
    pred = ["positive", "positive", "positive"]
    eval_step = EvaluationStep(true)
    result = eval_step.apply(pred)

    assert result["accuracy"] == pytest.approx(0.667, abs=0.001)
    assert result["precision"] == pytest.approx(0.667, abs=0.001)
    assert result["recall"] == pytest.approx(1.0, abs=0.001)
    assert result["f1"] == pytest.approx(0.8, abs=0.001)
