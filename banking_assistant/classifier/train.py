import os
import joblib
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import  accuracy_score


def build_models():
    vectorizer = TfidfVectorizer(max_features=10000,ngram_range=(1,2), stop_words='english')

    models = {
        "logreg": LogisticRegression(),
        "rf": RandomForestClassifier(),
        "svc": SVC()
    }

    param_grids = {
        "logreg": {
            "clf__C": [0.1, 1, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["liblinear"]
        },
        "rf": {
            "clf__n_estimators": [50, 100,200],
            "clf__max_depth": [None, 10, 20],
            "clf__criterion":["gini"]
        },
        "svc": {
            "clf__C": [0.1, 1],
            "clf__kernel": ["linear", "rbf"]
        }
    }

    pipelines = {
        name: Pipeline([
            ("vectorizer", vectorizer),
            ("clf", model)
        ])
        for name, model in models.items()
    }

    return pipelines, param_grids


def train_and_select_best(X, y, save_path: str) -> Tuple[str | None, Pipeline | None]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipelines, param_grids = build_models()

    best_model = None
    best_score = 0
    best_name = None

    for name, pipe in pipelines.items():
        print(f"\n🔍 Tuning model: {name}")
        grid = GridSearchCV(pipe, param_grids[name], scoring="f1_macro", cv=3)
        grid.fit(X_train, y_train)

        preds = grid.predict(X_test)
        score = accuracy_score(y_test, preds)
        print(f"{name} f1_macro: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_
            best_name = name

    if best_model:
        joblib.dump(best_model,save_path)
        print(f"\nBest model ({best_name}) saved to {save_path}")

    return best_name, best_model
