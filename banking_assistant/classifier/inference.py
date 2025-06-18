import joblib
from typing import List


def load_classifier(path: str):
    return joblib.load(path)


def predict(classifier, texts: List[str]) -> List[str]:
    return classifier.predict(texts)
