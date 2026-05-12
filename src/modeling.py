import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


def build_model_candidates() -> dict[str, Any]:
    """Return base estimators for tuning."""
    return {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=5000,
            class_weight='balanced',
            solver='liblinear',
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=-1,
        ),
    }


def build_param_grid() -> dict[str, dict[str, list[Any]]]:
    """Define hyperparameter grids for GridSearchCV."""
    return {
        'Logistic Regression': {
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga'],
        },
        'Decision Tree': {
            'max_depth': [5, 8, 12, None],
            'min_samples_leaf': [10, 20, 30],
            'max_features': ['sqrt', 'log2'],
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [8, 12, None],
            'min_samples_split': [4, 8],
            'max_features': ['sqrt', 'log2'],
        },
    }


def tune_models(X_train, y_train, cv: int = 5, scoring: str = 'f1') -> tuple[dict[str, Any], dict[str, GridSearchCV]]:
    """Tune model hyperparameters using GridSearchCV."""
    estimators = build_model_candidates()
    param_grid = build_param_grid()
    best_models: dict[str, Any] = {}
    search_results: dict[str, GridSearchCV] = {}

    for name, estimator in estimators.items():
        logger.info("Starting hyperparameter tuning for %s", name)
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid[name],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        search_results[name] = grid_search
        logger.info("Best params for %s: %s", name, grid_search.best_params_)
        logger.info("Best %s score for %s: %.4f", scoring, name, grid_search.best_score_)

    return best_models, search_results


def save_model_store(models: dict[str, Any], output_path: Path) -> None:
    """Save a dictionary of trained models to disk."""
    logger.info("Saving model artifacts to %s", output_path)
    joblib.dump(models, output_path)


def save_best_model(model: Any, path: Path) -> None:
    """Save the selected best model for production use."""
    logger.info("Saving best model to %s", path)
    joblib.dump(model, path)
