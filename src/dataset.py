"""
Script untuk persiapan Dataset dengan torch Dataset.
Pembuatan window
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.CONFIG import DATA_CONFIG, MODEL_CONFIG

class TimeSeriesDataset(Dataset):
    """
    Dataset untuk data time series (supervised)
    Args:
        - data_path (str) : path ke file csv.
        - window_size (int) : ukuran window.
        - stride (int) : langkah sliding window.
        - data_type (str) : train atau test.
        - scaler : scaler yang sudah di fit di train untuk test.
        - scale (bool) : scalin atau tidak.
    """
    def __init__(self, data_path: str, window_size: int, stride: int, data_type: str = 'train', scaler=None, scale: bool = True):
        self.data_type = data_type
        self.scale = scale

        # ambil ukuran window dan stride default dari config utk initial default
        self.window_size = window_size if window_size is not None else MODEL_CONFIG['window_size']
        self.stride = stride if stride is not None else MODEL_CONFIG['stride']

        # load data
        self.data = pd.read_csv(data_path)

        # cek kolom
        required_columns = ['value', 'label']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Dataset harus memiliki kolom '{col}'")

        # urutkan timestamo
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.sort_values(by = 'timestamp').reset_index(drop=True)
            self.timestamps = self.data['timestamp'].values
        else:
            self.timestamps = None

        # ekstrak value dan label
        self.values = self.data['value'].values.reshape(-1, 1) # [N, 1]
        self.labels = self.data['label'].values               

        print(f"Load {data_type} dari {data_path} dengan {len(self.data)} rows")

        # scaling
        if scale:
            # fit di train, pakai di test
            if scaler is None:
                self.scaler = StandardScaler()
                self.values = self.scaler.fit_transform(self.values)
                print(f"Fitting scaler")
            else:
                self.scaler = scaler
                self.values = self.scaler.transform(self.values)
                print(f"Pakai scaler yang sudah di fit")
        else:
            self.scaler = None

        # buat window
        self.windows, self.windows_labels, self.window_indices = self._create_windows()
        print(f"Panjang windows: {len(self.windows)} dengan window_size={self.window_size} dan stride={self.stride}")


    # helper untuk buat window
    def _create_windows(self):
        windows = []
        windows_labels = []
        window_indices = [] # menyimpan indeks awal setiap window

        for i in range(0, len(self.values) - self.window_size + 1, self.stride):
            window = self.values[i : i + self.window_size]
            label = self.labels[i + self.window_size - 1]  # label diambil dari titik terakhir window

            windows.append(window)
            windows_labels.append(label)
            window_indices.append(i + self.window_size - 1) # index di titik terakhir

        return np.array(windows), np.array(windows_labels), np.array(window_indices)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = torch.FloatTensor(self.windows[idx])
        label = torch.LongTensor([self.windows_labels[idx]])
        return window, label
    
    def get_scaler(self):
        return self.scaler
    
    def get_original_values(self):
        """Mengembalikan nilai original sebelum scalinf"""
        if self.scale and self.scaler is not None:
            return self.scaler.inverse_transform(self.values)
        return self.values
    
    def get_window_info(self, idx):
        """Informasi window"""
        return {
            "window" : self.windows[idx],
            "label" : self.windows_labels[idx],
            "original_index" : self.window_indices[idx],
            "timestamp" : self.timestamps[self.window_indices[idx]] if self.timestamps is not None else None
        }