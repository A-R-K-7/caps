import mlflow
import structlog
from configs.settings import settings

logger = structlog.get_logger()

class ExperimentTracker:
    def __init__(self, experiment_name: str = "caps_default"):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        
    def start_run(self, run_name: str):
        return mlflow.start_run(run_name=run_name)
        
    def log_params(self, params: dict):
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)
        
    def log_dict(self, dictionary: dict, artifact_file: str):
        mlflow.log_dict(dictionary, artifact_file)
