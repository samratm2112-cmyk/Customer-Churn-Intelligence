"""Visualization script for model evaluation and feature importance."""

from pathlib import Path

import joblib
from src.data_processing import load_data, preprocess_data, split_dataset
from src.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
)


def main() -> None:
    data_path = Path('data') / 'churn.csv'
    model_store_path = Path('model_store.pkl')

    if not data_path.exists():
        raise FileNotFoundError('Dataset not found. Run churn_prediction.py first.')
    if not model_store_path.exists():
        raise FileNotFoundError('Saved models not found. Run churn_prediction.py first.')

    df = load_data(data_path)
    X, y, _ = preprocess_data(df)
    _, X_test, _, y_test = split_dataset(X, y)

    model_store = joblib.load(model_store_path)
    report_models = {name: model for name, model in model_store.items() if name != 'best'}
    results = {}

    for name, model in report_models.items():
        metrics = compute_metrics(model, X_test, y_test)
        results[name] = metrics

    best_model = model_store['best']
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=best_model.predict(X_test),
        title='Confusion Matrix - Best Model',
        save_path=Path('confusion_matrix.png'),
    )

    if 'Random Forest' in report_models:
        plot_feature_importance(
            model=report_models['Random Forest'],
            feature_names=list(X.columns),
            title='Top Feature Importances - Random Forest',
            save_path=Path('feature_importance.png'),
        )
    else:
        plot_feature_importance(
            model=best_model,
            feature_names=list(X.columns),
            title='Feature Importance - Best Model',
            save_path=Path('feature_importance.png'),
        )

    plot_model_comparison(results, save_path=Path('model_comparison.png'))

    print('Saved confusion_matrix.png, feature_importance.png, model_comparison.png')


if __name__ == '__main__':
    main()
