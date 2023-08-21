
import os
import click
import pickle
# from typing import Any

import numpy as np
import scipy

# import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import mlflow

import warnings
# Ignore all warnings
# warnings.filterwarnings("ignore")
# Filter the specific warning message, MLflow autologging encountered a warning
# warnings.filterwarnings("ignore", category=UserWarning, module="setuptools")
warnings.filterwarnings("ignore", category=UserWarning, message="Setuptools is replacing distutils.")


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


def run_train(data_path="./output", dest_path="./model") -> None:
    """The main training pipeline""" 
    # Load train and test Data
    X_train, y_train = load_pickle("train.pkl", data_path)
    X_val, y_val     = load_pickle("val.pkl", data_path)
    # print(type(X_train), type(y_train))

    # MLflow settings
    # Build or Connect Database Offline
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # Connect Database Online
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Build or Connect mlflow experiment
    EXPERIMENT_NAME = "random-forest-train"
    mlflow.set_experiment(EXPERIMENT_NAME)
            
    # before your training code to enable automatic logging of sklearn metrics, params, and models
    # mlflow.sklearn.autolog()    
    
    with mlflow.start_run():
        # Optional: Set some information about Model
        mlflow.set_tag("developer", "muce")
        mlflow.set_tag("algorithm", "Machine Learning")
        mlflow.set_tag("train-data-path", f'{data_path}/train.pkl')
        mlflow.set_tag("valid-data-path", f'{data_path}/val.pkl')
        mlflow.set_tag("test-data-path",  f'{data_path}/test.pkl')        
        
        # Set Model params information
        params = {"max_depth": 9, "class_weight": "balanced", "random_state": 42}
        mlflow.log_params(params)
        
        # Build Model        
        rf     = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        
        # autolog_run = mlflow.last_active_run()

        # Log the validation Metric to the tracking server
        y_pred = rf.predict(X_val)
        print(classification_report(y_val, y_pred))

        pr_fscore   = precision_recall_fscore_support(y_val, y_pred, average='weighted')
        # Extract the F1-score from the tuple
        weighted_f1_score = pr_fscore[2]
        mlflow.log_metric("weighted_f1_score", weighted_f1_score)
        # print("weighted_f1_score", weighted_f1_score)
                        
        # Log Model two options
        # Option1: Just only model in log
        mlflow.sklearn.log_model(sk_model = rf, artifact_path = "model_mlflow")
                
        # Option 2: save Model, and Optional: Preprocessor or Pipeline in log
        # Create dest_path folder unless it already exists
        # pathlib.Path(dest_path).mkdir(exist_ok=True)
        os.makedirs(dest_path, exist_ok=True)
        local_path = os.path.join(dest_path, "ride_duration_rf_model.bin")
        with open(local_path, 'wb') as f_out:
            pickle.dump(rf, f_out)
            
        # whole proccess like pickle, saved Model, Optional: Preprocessor or Pipeline
        mlflow.log_artifact(local_path = local_path, artifact_path="model_pickle")        
        
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
    return None


if __name__ == '__main__':
    run_train()
