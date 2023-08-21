
import os
import pickle
import click
# from typing import Any

import numpy as np
import scipy

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import optuna
from optuna.samplers import TPESampler

import mlflow


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


def run_optimization(data_path="./output", num_trials=10) -> None:
    """The main optimization pipeline"""
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
    HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
    mlflow.set_experiment(HPO_EXPERIMENT_NAME)
    
    # before your training code to disable automatic logging of sklearn metrics, params, and models
    mlflow.sklearn.autolog(disable=True)

    # Optional: Set some information about Model
    mlflow.set_tag("developer", "muce")
    mlflow.set_tag("algorithm", "Machine Learning")
    mlflow.set_tag("train-data-path", f'{data_path}/train.pkl')
    mlflow.set_tag("valid-data-path", f'{data_path}/val.pkl')
    mlflow.set_tag("test-data-path",  f'{data_path}/test.pkl')

    def objective(trial):
        with mlflow.start_run(nested=True):
            # Define params            
            params = {
                "class_weight"     : trial.suggest_categorical(name='class_weight', choices=["balanced"]),
                'n_estimators'     : trial.suggest_int('n_estimators', 70, 120, 10),
                'max_depth'        : trial.suggest_int('max_depth', 3, 21, 2),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 17, 3),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 6, 1),
                'random_state'     : 42,
                'n_jobs'           : -1
            }            
            # Log the model params to the tracking server
            mlflow.log_params(params)
            
            # Build Model   
            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)
            
            # Log the validation Metric to the tracking server
            y_pred_val = rf.predict(X_val)
            # print(classification_report(y_val, y_pred))

            pr_fscore_val   = precision_recall_fscore_support(y_val, y_pred_val, average='weighted')
            # Extract the F1-score from the tuple
            weighted_f1_score_val = pr_fscore_val[2]
            mlflow.log_metric("weighted_f1_score_val", weighted_f1_score_val)
            # print("weighted_f1_score_val", weighted_f1_score_val)
        return weighted_f1_score_val

    sampler = TPESampler(seed=42)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(func = objective, n_trials=num_trials, n_jobs=-1)
    
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
    return None


if __name__ == '__main__':
    run_optimization()
