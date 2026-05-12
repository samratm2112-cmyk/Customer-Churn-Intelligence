import logging
from pathlib import Path

import joblib

from src.data_processing import (
    describe_class_balance,
    load_data,
    preprocess_data,
    split_dataset,
)
from src.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
)
from src.modeling import save_best_model, save_model_store, tune_models


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    data_path = Path('data') / 'churn.csv'
    model_store_path = Path('model_store.pkl')
    best_model_path = Path('best_model.pkl')
    preprocessor_path = Path('column_transformer.pkl')

    logger.info('Starting churn prediction training pipeline')
    df = load_data(data_path)

    X, y, preprocessor = preprocess_data(df)
    describe_class_balance(y)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    logger.info('Running hyperparameter tuning and model selection')
    best_models, search_results = tune_models(X_train, y_train)

    results: dict[str, dict] = {}
    model_store: dict[str, object] = {}

    for name, model in best_models.items():
        metrics = compute_metrics(model, X_test, y_test)
        results[name] = metrics
        model_store[name] = model
        logger.info('%s classification report:\n%s', name, metrics['classification_report'])

    best_model_name = max(results, key=lambda key: results[key]['accuracy'])
    best_model = best_models[best_model_name]
    model_store['best'] = best_model

    save_model_store(model_store, model_store_path)
    save_best_model(best_model, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info('Selected best model: %s', best_model_name)

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=best_model.predict(X_test),
        title=f'Confusion Matrix - {best_model_name}',
        save_path=Path('confusion_matrix.png'),
    )
    plot_feature_importance(
        model=best_model,
        feature_names=list(X.columns),
        title=f'Feature Importance - {best_model_name}',
        save_path=Path('feature_importance.png'),
    )
    plot_model_comparison(results, save_path=Path('model_comparison.png'))

    logger.info('Training pipeline finished successfully')


if __name__ == '__main__':
    main()
