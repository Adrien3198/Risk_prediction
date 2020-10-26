from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from evaluation import get_metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import data_preparation

from urllib.parse import urlparse

import mlflow
import mlflow.sklearn


def train():
    """
        Train and evaluate a random forest classifier model with mlflow tracking
    """
    df = data_preparation.preprocess()

    y = df["TARGET"]
    X = df.drop(columns="TARGET")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    with mlflow.start_run():
        
        max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else None
        min_samples_split = int(sys.argv[2]) if len(sys.argv) > 2 else 2

        model = RandomForestClassifier(n_jobs=-1)
        model.fit(X=X_train, y=y_train)
        y_pred_rfc = model.predict(X_test)

        accuracy, precision, recall, f1, support = get_metrics(y_test, y_pred_rfc)
        
        mlflow.log_params({
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        })

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision_0": precision[0],
            "precision_1": precision[1],
            "f1_0": f1[0],
            "f1_1": f1[1],
            "recall_0": recall[0],
            "recall_1": recall[1],
            "support_0": support[0],
            "support_1": support[1]
        })

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                model, "model_rfc", registered_model_name="RandomForestClassifier")
        else:
            mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    train()