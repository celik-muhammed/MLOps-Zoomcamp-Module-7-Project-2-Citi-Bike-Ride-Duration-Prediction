
#!/usr/bin/env python
# coding: utf-8

import os
import io
import sys
import pickle
import zipfile
import requests

import numpy as np
import pandas as pd

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
filepath = os.path.join(raw_data_path, csv_filename)

# Extract the CSV file from the ZIP file
with zipfile.ZipFile(zip_data, 'r') as zip_ref:
    zip_ref.extract(csv_filename, path=raw_data_path)  # Extract to a specific directory

categorical_features = [
    'start_station_id',
    'end_station_id'
]


with open('model/lin_reg_model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)
    
    
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


df = read_data(filepath)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


dicts      = df[categorical_features].to_dict(orient='records')
X_val      = dv.transform(dicts)
y_pred_val = lr.predict(X_val)

print('predicted mean duration:', y_pred_val.mean().round(2))


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred_val


os.makedirs('output', exist_ok=True)
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
