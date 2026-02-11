"""MLflow tracking utilities for experiment management."""

import os
from contextlib import contextmanager
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


class MLflowTracker:
    """MLflow experiment tracker for ML pipelines.

    Usage:
        tracker = MLflowTracker(experiment_name="heart-disease")
        with tracker.start_run(run_name="catboost-v1"):
            tracker.log_params({"learning_rate": 0.1, "depth": 6})
            tracker.log_metrics({"auc": 0.95, "accuracy": 0.88})
            tracker.log_model(model, "model")
    """

    def __init__(
        self,
        experiment_name: str = "heart-disease-prediction",
        tracking_uri: str | None = None,
    ):
        """Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server (default: local ./mlruns)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")
        self._run = None

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        self.client = MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None):
        """Start an MLflow run context.

        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
        """
        with mlflow.start_run(run_name=run_name) as run:
            self._run = run
            if tags:
                mlflow.set_tags(tags)
            yield run
            self._run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Args:
            params: Dictionary of parameter names and values
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for tracking over time
        """
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact file.

        Args:
            local_path: Path to local file
            artifact_path: Optional path within artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model: Any, artifact_path: str, model_type: str = "sklearn") -> None:
        """Log a model to MLflow.

        Args:
            model: Model object to log
            artifact_path: Path for the model artifact
            model_type: Type of model (sklearn, catboost, lightgbm, xgboost)
        """
        if model_type == "catboost":
            mlflow.catboost.log_model(model, artifact_path)
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path)
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path)
        else:
            mlflow.sklearn.log_model(model, artifact_path)

    def log_feature_importance(
        self,
        feature_names: list[str],
        importance_values: list[float],
    ) -> None:
        """Log feature importance as a table artifact.

        Args:
            feature_names: List of feature names
            importance_values: List of importance values
        """
        import pandas as pd

        df = pd.DataFrame({"feature": feature_names, "importance": importance_values})
        df = df.sort_values("importance", ascending=False)

        # Save as CSV artifact
        artifact_path = "feature_importance.csv"
        df.to_csv(artifact_path, index=False)
        self.log_artifact(artifact_path)
        os.remove(artifact_path)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag key
            value: Tag value
        """
        mlflow.set_tag(key, value)

    @property
    def run_id(self) -> str | None:
        """Get current run ID."""
        return self._run.info.run_id if self._run else None


def get_best_run(experiment_name: str, metric: str = "auc") -> dict[str, Any]:
    """Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize (default: auc)

    Returns:
        Dictionary with run info, params, and metrics
    """
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        return {}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if not runs:
        return {}

    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "params": best_run.data.params,
        "metrics": best_run.data.metrics,
        "tags": best_run.data.tags,
    }
