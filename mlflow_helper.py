import os
import sys
import io
import mlflow
import subprocess
from nbconvert import HTMLExporter
import nbformat

class MLFlowLogger:
    """
    A logger class for MLFlow to manage experiments and log parameters, metrics, and artifacts.

    Attributes:
        experiment_name (str): The name of the MLFlow experiment.
        run_name (str|None): The name of the current MLFlow run.
    """

    def __init__(self, experiment_name: str = "Default", run_name: str | None = None):
        """
        Initializes the MLFlowLogger with specified experiment and run names.

        Sets up MLFlow tracking and S3 endpoint environment variables, and ensures AWS credentials are configured.

        Args:
            experiment_name (str, optional): Name of the experiment. Defaults to "Default".
            run_name (str | None, optional): Name of the run. Defaults to None.
        """
        self.MLFLOW_SERVER_URL = "MLFLOW_SERVER_URL"
        # mlflow credentials
        self._user = "MLFLOW_USER"
        self._password= "MLFLOW_PASSWORD"
        
        # s3 endpoint for artifacts
        self._s3_endpoint = "URL_S3_ENDPOINT"
        self._aws_access_key_id = "minio"
        self._aws_secret_access_key = "_aws_secret_access_key"
        self._bucket_name = "_bucket_name"

        self.set_env_variables()
        
        self.experiment_name = experiment_name
        self.run_name = run_name


    def set_env_variables(self):
        """
        Sets the necessary environment variables for MLFlow and AWS S3 integration.
        
        Checks if AWS credentials file exists and creates it if not present.
        """
        os.environ["MLFLOW_TRACKING_USERNAME"] = self._user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self._password
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self._s3_endpoint
        os.environ["AWS_BUCKET_NAME"] = self._bucket_name
        
        # check if credentials for artifacts are set
        os.makedirs(os.path.expanduser("~/.aws"), exist_ok=True)

        if not os.path.isfile(os.path.expanduser("~/.aws/credentials")):
            content = f"""
            [default]
            aws_access_key_id={self._aws_access_key_id}
            aws_secret_access_key={self._aws_secret_access_key}
            """
            with open(os.path.expanduser("~/.aws/credentials"), "w") as f:
                f.write(content)

    @staticmethod
    def log_commit_hash(logger = None, run_id: str | None = None):
        """
        Logs the current Git commit hash to MLFlow.

        Args:
            logger (MLFlowLogger, optional): An instance of MLFlowLogger. Defaults to None.
            run_id (str | None, optional): The run ID for logging. Defaults to None.
        """
        # Get the current git commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

        # Log the commit hash
        if logger is None:
            mlflow.log_param("git_commit_hash", commit_hash)
        else:
            logger.experiment.log_param(run_id=run_id, key="git_commit_hash", value=commit_hash)

    @staticmethod
    def log_notebook(filename: str, artifact_name: str = "notebook.html", logger = None, run_id: str | None = None):
        """
        Converts a Jupyter Notebook to HTML and logs it as an artifact in MLFlow.

        Args:
            filename (str): The filename of the Jupyter Notebook.
            artifact_name (str, optional): The name for the logged artifact. Defaults to "notebook.html".
            logger (MLFlowLogger, optional): An instance of MLFlowLogger. Defaults to None.
            run_id (str | None, optional): The run ID for logging. Defaults to None.
        """
        # Load the current notebook
        current_notebook = nbformat.read(open(filename), as_version=4)

        # Export the notebook to HTML
        exporter = HTMLExporter()
        body, _ = exporter.from_notebook_node(current_notebook)

        # Save the HTML to a file
        with open(artifact_name, "w") as file:
            file.write(body)
        if logger is None:
            mlflow.log_artifact(artifact_name, artifact_path="notebook")
        else:
            logger.experiment.log_artifact(run_id=run_id, local_path=artifact_name, artifact_path="notebook")
        os.remove(artifact_name)



    def start_run(self):
        """
        Starts an MLFlow run, setting up the tracking URI and experiment.
        """
        mlflow.set_tracking_uri(self.MLFLOW_SERVER_URL)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run()

    def end_run(self):
        """
        Ends the current MLFlow run.
        """
        mlflow.end_run()
        


if __name__ == "__main__":
    mlflow_logger = MLFlowLogger()
    mlflow_logger.start_run()
    mlflow_logger.log_commit_hash()
    mlflow_logger.end_run()