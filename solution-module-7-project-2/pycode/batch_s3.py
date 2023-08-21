
import os
import io
import sys
import pickle
import zipfile
import requests

import numpy as np
import pandas as pd


def load_model(file_path) -> tuple:
    with open(file_path, 'rb') as f_in:
        return pickle.load(f_in)


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


def process_data(input_file, year, month) -> pd.DataFrame:
    df = read_data(input_file)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df


def predict_duration(df: pd.DataFrame, dv, lr) -> np.ndarray:
    """Predict the duration using the trained model"""
    dicts  = df[categorical_features].to_dict(orient='records')
    X_val  = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    return y_pred


def save_results(df: pd.DataFrame, y_pred: np.ndarray, output_file: str) -> None:
    """Save the predicted results to a parquet file"""
    os.makedirs('output', exist_ok=True)
    
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    df_result.to_parquet(        
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    return None


def main() -> None:
    # Step 1: Loading model
    dv, lr = load_model(model_file_path)

    # Step 2: Reading data
    df = process_data(input_file, year, month)
        
    # Step 3: Predict data
    y_pred = predict_duration(df, dv, lr)

    # Print Prediction
    print('predicted mean duration:', y_pred.mean().round(2))

    # save_results
    save_results(df, y_pred, output_file)
    return None

    
if __name__ == '__main__':
    # Global Parameters
    location = "JC-"
    year     = int(sys.argv[1]) # 2023
    month    = int(sys.argv[2]) # 5

    # output
    output_file = f'output/{location}tripdata_{year:04d}-{month:02d}.parquet'

    # input
    raw_data_path='data/'
    # URL of the ZIP file
    filename = f'{location}{year}{month:0>2}-citibike-tripdata.csv.zip'
    zip_file_url = f'https://s3.amazonaws.com/tripdata/{filename}'

    # Download the ZIP file from the URL
    response = requests.get(zip_file_url)
    zip_data = io.BytesIO(response.content)

    # CSV filename within the ZIP file
    csv_filename = filename.replace('.zip', '')
    input_file = os.path.join(raw_data_path, csv_filename)

    # Extract the CSV file from the ZIP file
    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        zip_ref.extract(csv_filename, path=raw_data_path)  # Extract to a specific directory

    categorical_features = [
        'start_station_id',
        'end_station_id'
    ]
    model_file_path = 'model/lin_reg_model.bin' 

    main()
