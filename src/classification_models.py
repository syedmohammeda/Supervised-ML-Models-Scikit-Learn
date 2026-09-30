"""Classification models used in the supervised learning project."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(X_train, y_train, random_state: int = 42, max_iter: int = 1000):
    """Train a logistic regression classifier with feature scaling."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=random_state, max_iter=max_iter)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_decision_tree_classifier(X_train, y_train, random_state: int = 42):
    """Train and return a decision tree classifier."""
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest_classifier(X_train, y_train, random_state: int = 42, n_estimators: int = 200):
    """Train and return a random forest classifier."""
    model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model


def train_knn_classifier(X_train, y_train, random_state: int = 42, n_neighbors: int = 5):
    """Train a k-nearest neighbors classifier with feature scaling."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline
