"""Regression models used in the supervised learning project."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def train_linear_regression(X_train, y_train):
    """Train and return a LinearRegression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree_regressor(X_train, y_train, random_state: int = 42):
    """Train and return a DecisionTreeRegressor model."""
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest_regressor(X_train, y_train, random_state: int = 42, n_estimators: int = 200):
    """Train and return a RandomForestRegressor model."""
    model = RandomForestRegressor(random_state=random_state, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model
