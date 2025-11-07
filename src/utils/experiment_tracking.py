"""
Experiment tracking integration for MLflow and Weights & Biases.

Provides unified interface for tracking experiments, metrics, and artifacts.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracker supporting MLflow and Weights & Biases."""

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        use_mlflow: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            run_name: Name of this specific run
            tracking_uri: MLflow tracking URI
            use_mlflow: Whether to use MLflow
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity/team name
            config: Configuration dictionary to log
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.config = config or {}

        # MLflow setup
        if self.use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow

                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)

                mlflow.set_experiment(experiment_name)

                # Start run
                self.mlflow_run = mlflow.start_run(run_name=run_name)
                logger.info(f"MLflow tracking initialized: {mlflow.get_tracking_uri()}")
                logger.info(f"Run ID: {self.mlflow_run.info.run_id}")

                # Log config
                if config:
                    mlflow.log_params(self._flatten_dict(config))

            except ImportError:
                logger.warning("MLflow not installed. Install with: pip install mlflow")
                self.use_mlflow = False
            except Exception as e:
                logger.error(f"Failed to initialize MLflow: {e}")
                self.use_mlflow = False

        # Weights & Biases setup
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb

                self.wandb_run = wandb.init(
                    project=wandb_project or experiment_name,
                    entity=wandb_entity,
                    name=run_name,
                    config=config,
                )
                logger.info(f"W&B tracking initialized: {wandb.run.get_url()}")

            except ImportError:
                logger.warning("Weights & Biases not installed. Install with: pip install wandb")
                self.use_wandb = False
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

        if not self.use_mlflow and not self.use_wandb:
            logger.warning("No experiment tracking enabled!")

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary for logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to tracking systems.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
        """
        if self.use_mlflow:
            try:
                for name, value in metrics.items():
                    self.mlflow.log_metric(name, value, step=step)
            except Exception as e:
                logger.error(f"Failed to log metrics to MLflow: {e}")

        if self.use_wandb:
            try:
                log_dict = metrics.copy()
                if step is not None:
                    log_dict['step'] = step
                self.wandb.log(log_dict)
            except Exception as e:
                logger.error(f"Failed to log metrics to W&B: {e}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters/hyperparameters.

        Args:
            params: Dictionary of parameter names and values
        """
        if self.use_mlflow:
            try:
                self.mlflow.log_params(self._flatten_dict(params))
            except Exception as e:
                logger.error(f"Failed to log params to MLflow: {e}")

        if self.use_wandb:
            try:
                self.wandb.config.update(params)
            except Exception as e:
                logger.error(f"Failed to log params to W&B: {e}")

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """
        Log file as artifact.

        Args:
            file_path: Path to file to log
            artifact_path: Optional destination path in artifact store
        """
        if self.use_mlflow:
            try:
                self.mlflow.log_artifact(file_path, artifact_path)
            except Exception as e:
                logger.error(f"Failed to log artifact to MLflow: {e}")

        if self.use_wandb:
            try:
                artifact = self.wandb.Artifact(
                    name=Path(file_path).stem,
                    type='model' if file_path.endswith('.pt') else 'file'
                )
                artifact.add_file(file_path)
                self.wandb.log_artifact(artifact)
            except Exception as e:
                logger.error(f"Failed to log artifact to W&B: {e}")

    def log_model(self, model_path: str, model_name: str = "model"):
        """
        Log trained model.

        Args:
            model_path: Path to saved model file
            model_name: Name for the model
        """
        if self.use_mlflow:
            try:
                self.mlflow.log_artifact(model_path, f"models/{model_name}")
            except Exception as e:
                logger.error(f"Failed to log model to MLflow: {e}")

        if self.use_wandb:
            try:
                artifact = self.wandb.Artifact(model_name, type='model')
                artifact.add_file(model_path)
                self.wandb.log_artifact(artifact)
            except Exception as e:
                logger.error(f"Failed to log model to W&B: {e}")

    def log_dict(self, dictionary: Dict[str, Any], filename: str):
        """
        Log dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to save
            filename: Filename for JSON file
        """
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dictionary, f, indent=2)
            temp_path = f.name

        self.log_artifact(temp_path, filename)
        os.unlink(temp_path)

    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the run.

        Args:
            tags: Dictionary of tag names and values
        """
        if self.use_mlflow:
            try:
                for key, value in tags.items():
                    self.mlflow.set_tag(key, value)
            except Exception as e:
                logger.error(f"Failed to set MLflow tags: {e}")

        if self.use_wandb:
            try:
                self.wandb.config.update({"tags": tags})
            except Exception as e:
                logger.error(f"Failed to set W&B tags: {e}")

    def finish(self):
        """End the tracking run."""
        if self.use_mlflow:
            try:
                self.mlflow.end_run()
                logger.info("MLflow run ended")
            except Exception as e:
                logger.error(f"Failed to end MLflow run: {e}")

        if self.use_wandb:
            try:
                self.wandb.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


class MLflowTracker:
    """Simplified MLflow-only tracker."""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """Initialize MLflow tracker."""
        try:
            import mlflow
            self.mlflow = mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run()
            logger.info(f"MLflow initialized: {mlflow.get_tracking_uri()}")

        except ImportError:
            logger.error("MLflow not installed. Install with: pip install mlflow")
            raise

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        for name, value in metrics.items():
            self.mlflow.log_metric(name, value, step=step)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        self.mlflow.log_params(params)

    def log_artifact(self, file_path: str):
        """Log artifact."""
        self.mlflow.log_artifact(file_path)

    def finish(self):
        """End run."""
        self.mlflow.end_run()


class WandbTracker:
    """Simplified Weights & Biases-only tracker."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize W&B tracker."""
        try:
            import wandb
            self.wandb = wandb

            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
            )
            logger.info(f"W&B initialized: {wandb.run.get_url()}")

        except ImportError:
            logger.error("Weights & Biases not installed. Install with: pip install wandb")
            raise

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics."""
        self.wandb.log(metrics)

    def log_artifact(self, file_path: str, name: str, type: str = "file"):
        """Log artifact."""
        artifact = self.wandb.Artifact(name, type=type)
        artifact.add_file(file_path)
        self.wandb.log_artifact(artifact)

    def finish(self):
        """Finish run."""
        self.wandb.finish()


# Example usage
if __name__ == '__main__':
    # Example 1: Use both MLflow and W&B
    config = {
        'model': 'gemma-2b',
        'batch_size': 16,
        'learning_rate': 2e-5,
    }

    with ExperimentTracker(
        experiment_name="gemma_redsm5",
        run_name="test_run",
        use_mlflow=True,
        use_wandb=False,  # Set to True to enable W&B
        config=config,
    ) as tracker:
        # Log metrics during training
        for epoch in range(5):
            metrics = {
                'train_loss': 0.5 - epoch * 0.05,
                'val_loss': 0.6 - epoch * 0.04,
                'val_f1': 0.7 + epoch * 0.03,
            }
            tracker.log_metrics(metrics, step=epoch)

        # Log final model
        # tracker.log_model("model.pt", "best_model")

    # Example 2: MLflow only
    with MLflowTracker("gemma_redsm5") as tracker:
        tracker.log_params({'lr': 2e-5})
        tracker.log_metrics({'loss': 0.5})

    print("Experiment tracking example completed")
