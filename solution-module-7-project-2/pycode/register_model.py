
import os
import click
import pickle
# from typing import Any

import numpy as np
import scipy

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

import warnings
# Ignore all warnings
# warnings.filterwarnings("ignore")
# Filter the specific warning message, MLflow autologging encountered a warning
# warnings.filterwarnings("ignore", category=UserWarning, module="setuptools")
warnings.filterwarnings("ignore", category=UserWarning, message="Setuptools is replacing distutils.")
warnings.filterwarnings("ignore", category=UserWarning, message="Distutils was imported before Setuptools,*")


def load_pickle(
    filename: str, data_path: str
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        np.ndarray
    ]
):
    file_path = os.path.join(data_path, filename)
    with open(file_path, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params, experiment_name): 
    """The main training pipeline"""    
    # Load train, val and test Data
    X_train, y_train = load_pickle("train.pkl", data_path)
    X_val, y_val     = load_pickle("val.pkl", data_path)
    X_test, y_test   = load_pickle("test.pkl", data_path)
    # print(type(X_train), type(y_train))
    
    # MLflow settings
    # Build or Connect Database Offline
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # Connect Database Online
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Build or Connect mlflow experiment
    EXPERIMENT_NAME = experiment_name
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # before your training code to enable automatic logging of sklearn metrics, params, and models
    # mlflow.sklearn.autolog()

    with mlflow.start_run(nested=True):
        # Optional: Set some information about Model
        mlflow.set_tag("developer", "muce")
        mlflow.set_tag("algorithm", "Machine Learning")
        mlflow.set_tag("train-data-path", f'{data_path}/train.pkl')
        mlflow.set_tag("valid-data-path", f'{data_path}/val.pkl')
        mlflow.set_tag("test-data-path",  f'{data_path}/test.pkl')  

        # Set Model params information
        RF_PARAMS = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs']
        for param in RF_PARAMS:
            params[param] = int(params[param])
            
        # Log the model params to the tracking server
        mlflow.log_params(params)

        # Build Model
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        # Log the validation and test Metric to the tracking server
        y_pred_val = rf.predict(X_val)
        y_pred_test = rf.predict(X_test)
        pr_fscore_val   = precision_recall_fscore_support(y_val, y_pred_val, average='weighted')
        pr_fscore_test   = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
        # Extract the F1-score from the tuple
        weighted_f1_score_val = pr_fscore_val[2]
        weighted_f1_score_test = pr_fscore_test[2]
        mlflow.log_metric("weighted_f1_score_val", weighted_f1_score_val)
        mlflow.log_metric("weighted_f1_score_test", weighted_f1_score_test)
        # print("weighted_f1_score_val", weighted_f1_score_val)

        # Log the model
        # Option1: Just only model in log
        mlflow.sklearn.log_model(sk_model = rf, artifact_path = "model_mlflow")
        
        # print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
    return None


def run_register_model(data_path="./output", top_n=5) -> None:
    """The main optimization pipeline"""
    # Parameters
    EXPERIMENT_NAME = "random-forest-best-models"
    HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
    client = MlflowClient("sqlite:///mlflow.db")

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.weighted_f1_score_val DESC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params, experiment_name=EXPERIMENT_NAME)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.weighted_f1_score_test DESC"]
    )[0]

    # Register the best model
    run_id     = best_run.info.run_id
    model_uri  = f"runs:/{run_id}/model"
    model_name = "rf-best-model"
    mlflow.register_model(model_uri, name=model_name)

    # print("Test weighted_f1_score_test of the best model: {:.4f}".format(best_run.data.metrics["weighted_f1_score_test"]))
    return None


if __name__ == '__main__':
    run_register_model()
