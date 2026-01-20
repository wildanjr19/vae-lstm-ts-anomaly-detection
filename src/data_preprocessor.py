"""
Script untuk praproses data : splitting data
Dataset diharapkan memiliki kolom 'timestamp', 'value', dan 'label'
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.CONFIG import DATA_CONFIG, PATHS, print_config

def prepare_data():
    "Fungsi untuk splitting data, train (normal) dan test (normal + anomali)"
    # buat folder
    os.makedirs(PATHS["data"]["processed"], exist_ok=True)
    os.makedirs(PATHS["outputs"]["models"], exist_ok=True)
    os.makedirs(PATHS["outputs"]["figures"], exist_ok=True)

    # load data
    print(f"Load data from {DATA_CONFIG['data_path']}")
    data = pd.read_csv(DATA_CONFIG['data_path'])

    # konversi timestamp ke format datetime
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values(by = 'timestamp').reset_index(drop=True)

    # cek kolom label
    if 'label' not in data.columns:
        print("Error: dataset harus memiliki kolom label (supervised setting)")
        return None, None
    
    # pisahkan normal dan anomali
    normal_data = data[data['label'] == 0]
    anomaly_data = data[data['label'] == 1]

    print(f"Normal data : {len(normal_data)} rows")
    print(f"Anomaly data : {len(anomaly_data)} rows")

    # split data train dan test
    split_idx = int(len(normal_data) * DATA_CONFIG['train_normal_ratio'])

    train_normal = normal_data.iloc[:split_idx].copy()
    test_normal = normal_data.iloc[split_idx:].copy()

    # test = normal + anomali
    test_data =pd.concat([test_normal, anomaly_data])
    if 'timestamp' in test_data.columns:
        test_data = test_data.sort_values(by = 'timestamp').reset_index(drop=True)

    print(f"Train set : {len(train_normal)} normal only")
    print(f"Test set : {len(test_data)} normal + anomaly")

    # save processed data
    train_path = os.path.join(PATHS["data"]["processed"], "train_normal.csv")
    test_path = os.path.join(PATHS["data"]["processed"], "test_data.csv")
    
    train_normal.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Saved train data to {train_path}")
    print(f"Saved test data to {test_path}")

    return train_path, test_path

if __name__ == "__main__":
    print_config()
    train_path, test_path = prepare_data()
    print("Data preparation selesai.")