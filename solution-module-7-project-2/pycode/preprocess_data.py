
import os
import pickle
import zipfile
import requests
# import pathlib
# import urllib.request
from glob import glob
# from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


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


def preprocess(df: pd.DataFrame, dv: DictVectorizer = None, fit_dv: bool = False):
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


def dump_pickle(obj, filename: str, dest_path: str): 
    file_path = os.path.join(dest_path, filename)
       
    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)
    with open(file_path, "wb") as f_out:
        return pickle.dump(obj, f_out)              
                

def run_data_prep(raw_data_path="./data", dest_path="./output", location="JC-", years="2023", months="5 6 7") -> None:  
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

    # Save DictVectorizer and datasets
    dump_pickle(dv, "dv.pkl", dest_path)
    dump_pickle((X_train, y_train), "train.pkl", dest_path)
    dump_pickle((X_val, y_val), "val.pkl", dest_path)
    dump_pickle((X_test, y_test), "test.pkl", dest_path)


if __name__ == '__main__':
    run_data_prep()
