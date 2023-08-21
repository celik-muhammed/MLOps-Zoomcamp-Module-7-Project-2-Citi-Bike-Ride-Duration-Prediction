
import os
import pickle
import zipfile

# import pathlib
import requests
import urllib.request
from glob import glob
from datetime import date
from datetime import timedelta

import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import mlflow
import prefect
from prefect import task, flow
from prefect.tasks import task_input_hash
from prefect.artifacts import create_markdown_artifact

# from prefect_aws import S3Bucket
# from prefect_email import EmailServerCredentials, email_send_message

import warnings
# Ignore all warnings
# warnings.filterwarnings("ignore")
# Filter the specific warning message, MLflow autologging encountered a warning
# warnings.filterwarnings("ignore", category=UserWarning, module="setuptools")
warnings.filterwarnings("ignore", category=UserWarning, message="Setuptools is replacing distutils.")


@task(name="Fetch Data", cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1),
      retries=3, log_prints=True, )
def fetch_data(raw_data_path: str, location: str, year: int, month: int) -> None:
    """Fetches data from the NYC Taxi dataset and saves it locally"""
    filename = f'{location}{year}{month:0>2}-citibike-tripdata.csv.zip'
    filepath = os.path.join(raw_data_path, filename)
    url      = f'https://s3.amazonaws.com/tripdata/{filename}'

    # Create dest_path folder unless it already exists
    os.makedirs(raw_data_path, exist_ok=True)
    
    # Download the data from the NYC Taxi dataset
    # os.system(f"wget -q -N -P {raw_data_path} {url}")
    # urllib.request.urlretrieve(url, filename)
    response = requests.get(url)
    with open(filepath, "wb") as f_out:
        f_out.write(response.content)
    
    # Extract the CSV file from the ZIP file
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extract(filename.replace('.zip', ''), path=raw_data_path)  # Extract to a specific directory
    return None


@flow(name="Data Downloads", log_prints=True)
def download_data(raw_data_path: str, locations: list, years: list, months: list) -> None:
    try:
        # Download the data from the NYC Taxi dataset    
        for loc in locations: 
            for year in years:       
                for month in months:
                    fetch_data(raw_data_path, loc, year, month)
    except Exception as e:
        print("In download_data Something Wrong...", e)
        pass
    return None
    
    
@task(name="Read Data", retries=3, retry_delay_seconds=2, log_prints=None)
def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_csv(filename)

    # Convert Datetime
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at']   = pd.to_datetime(df['ended_at'])

    # Calculate duration, Convert duration to minutes
    df['duration'] = df['ended_at'] - df['started_at']
    df['duration_minutes'] = df['duration'].dt.total_seconds() / 60

    # Define criteria for outliers 
    lower_threshold = 1   
    upper_threshold = 60  

    # Filter DataFrame based on outlier criteria
    df = df[
        (df['duration_minutes'] >= lower_threshold) & 
        (df['duration_minutes'] <= upper_threshold)
    ]

    # Define the categorical columns
    categorical_features = [
        'start_station_id',
        'end_station_id'
    ]
    df[categorical_features] = df[categorical_features].astype(str)
    # print(df.shape)
    return df


@task(name="Preprocess: Add Features", log_prints=True)
def preprocess(
    df: pd.DataFrame,dv: DictVectorizer = None, fit_dv: bool = False
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    def haversine_distance(row):
        lat1, lon1, lat2, lon2 = row['start_lat'], row['start_lng'], row['end_lat'], row['end_lng']
        # Convert latitude and longitude from degrees to radians
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

        # Radius of the Earth in kilometers
        radius = 6371.0

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = radius * c    
        return distance

    """Add features to the model"""
    # Add location ID
    df['start_to_end_station_id'] = df['start_station_id'] + '_' + df['end_station_id']
    categorical = ["start_to_end_station_id"]

    # Calc Distance
    df['trip_distance'] = df.apply(haversine_distance, axis=1).fillna(0)
    numerical   = ['trip_distance']
    dicts       = df[categorical + numerical].to_dict(orient='records')

    if fit_dv:
        # return sparse matrix
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
        
    # Convert X the sparse matrix  to pandas DataFrame, but too slow
    # X = pd.DataFrame(X.toarray(), columns=dv.get_feature_names_out())
    # X = pd.DataFrame.sparse.from_spmatrix(X, columns=dv.get_feature_names_out())

    try:
        # Extract the target
        target = 'member_casual'
        y = df[target].values
    except Exception as e:
        print("In preprocess Something Wrong...", e)
        pass
    # print(X.shape, y.shape)
    return (X, y), dv


@task(name="Train Best Model", log_prints=True)
def train_best_model(
    X_train  : scipy.sparse._csr.csr_matrix,
    X_val    : scipy.sparse._csr.csr_matrix,
    y_train  : np.ndarray,
    y_val    : np.ndarray,
    dv       : sklearn.feature_extraction.DictVectorizer,
    raw_data_path: str,
    dest_path: str,
) -> None:
    """train a model with best hyperparams and write everything out"""   
    # Assuming 'y' is your categorical label data
    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train)
    y_val_num = le.transform(y_val)

    # Load train and test Data
    train = xgb.DMatrix(X_train, label=y_train_num)
    valid = xgb.DMatrix(X_val, label=y_val_num)
    # print(type(X_train), type(y_train))

    # MLflow settings
    # Build or Connect Database Offline
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # Connect Database Online
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Build or Connect mlflow experiment
    EXPERIMENT_NAME = "nyc-taxi-experiment"
    mlflow.set_experiment(EXPERIMENT_NAME)

    # before your training code to enable automatic logging of sklearn metrics, params, and models
    # mlflow.xgboost.autolog()
    
    with mlflow.start_run(nested=True):
        # Optional: Set some information about Model
        mlflow.set_tag("developer", "muce")
        mlflow.set_tag("algorithm", "Machine Learning")
        mlflow.set_tag("train-data-path", f'{raw_data_path}/green_tripdata_2023-01.parquet')
        mlflow.set_tag("valid-data-path", f'{raw_data_path}/green_tripdata_2023-02.parquet')
        mlflow.set_tag("test-data-path",  f'{raw_data_path}/green_tripdata_2023-03.parquet') 
        
        # Set Model params information
        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            'objective': 'binary:logistic',
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }
        mlflow.log_params(best_params)

        # Build Model   
        booster = xgb.train(
            params               = best_params,
            dtrain               = train,
            num_boost_round      = 100,
            evals                = [(valid, "validation")],
            early_stopping_rounds=20,
        )
        # Log the validation Metric to the tracking server
        y_pred_val = booster.predict(valid)
        threshold = 0.5  # Example threshold value
        y_pred_class = (y_pred_val > threshold).astype(int)
        print(classification_report(y_val, le.inverse_transform(y_pred_class)))

        pr_fscore_val   = precision_recall_fscore_support(y_val, le.inverse_transform(y_pred_class), average='weighted')
        # Extract the F1-score from the tuple
        weighted_f1_score_val = pr_fscore_val[2]
        mlflow.log_metric("weighted_f1_score_val", weighted_f1_score_val)
        # print("weighted_f1_score", weighted_f1_score)              

        # Log Model two options
        # Option1: Just only model in log
        mlflow.xgboost.log_model(booster, artifact_path="model_mlflow")        
        
        # Option 2: save Model, and Optional: Preprocessor or Pipeline in log         
        # Create dest_path folder unless it already exists
        # pathlib.Path(dest_path).mkdir(exist_ok=True) 
        os.makedirs(dest_path, exist_ok=True)       
        local_file = os.path.join(dest_path, "preprocessor.b")
        with open(local_file, "wb") as f_out:
            pickle.dump(dv, f_out)
            
        # whole proccess like pickle, saved Model, Optional: Preprocessor or Pipeline
        mlflow.log_artifact(local_path = local_file, artifact_path="preprocessor")        
        
        # print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
    return None               


@flow(name="Main Flow")
def main_flow(raw_data_path="./data", dest_path="./output", location="JC-", years="2023", months="5 6 7") -> None:  
    # parameters
    locations = location.split(',')
    years     = [int(year) for year in years.split()]
    months    = [int(month) for month in months.split()]
    # print(locations, years, months)

    # Download data  
    download_data(raw_data_path, locations, years, months)
    # print(sorted(glob(f'./data/*')))
    
    # Load csv files
    df_train = read_data(
        os.path.join(raw_data_path, f'{locations[0]}{years[0]}{months[0]:0>2}-citibike-tripdata.csv')
    )
    df_val = read_data(
        os.path.join(raw_data_path, f'{locations[0]}{years[0]}{months[1]:0>2}-citibike-tripdata.csv')
    )
    df_test = read_data(
        os.path.join(raw_data_path, f'{locations[0]}{years[0]}{months[2]:0>2}-citibike-tripdata.csv')
    )

    # Fit the DictVectorizer and preprocess data
    (X_train, y_train), dv = preprocess(df_train, fit_dv=True)
    (X_val, y_val)    , _  = preprocess(df_val, dv)
    (X_test, y_test)  , _  = preprocess(df_test, dv)

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv, raw_data_path, dest_path)
    return None   


if __name__ == "__main__":
    main_flow()
