# src/model.py

from sklearn.pipeline import Pipeline

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)



def logistic_model(preprocessor):

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])


def decision_tree_model(preprocessor):

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            random_state=42
        ))
    ])


def random_forest_model(preprocessor):

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])


def gradient_boosting_model(preprocessor):

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(
            random_state=42
        ))
    ])


def adaboost_model(preprocessor):

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", AdaBoostClassifier(
            random_state=42
        ))
    ])


