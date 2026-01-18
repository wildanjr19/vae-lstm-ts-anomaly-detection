"""
Config untuk penelitian VAE LSTM Anomaly Detection
Confg meliputi : Data, Model, Training, Output Path, dan lain-lain
"""

import torch

# CONFIG DATA
DATA_CONFIG = {
    "data_path" : "../data/speed_7578_labeled.csv",
    "train_normal_path" : "data/processed/train_normal.csv",    # normal
    "test_data_path" : "data/processed/test_data.csv",          # normal + anomali
    "window_size" : 10,
    "stride" : 1,
    "train_normal_rato" : 0.8,
    "test_size" : 0.2,
    "random_seed" : 42
}

# MODEL CONFIG
MODEL_CONFIG = {
    "input_dim" : 1,
    "hidden_dim" : 64,
    "latent_dim" : 16,
    "lstm_layers" : 2,
    "dropout" : 0.2,
    "beta" : 1.0
}

# TRAINING CONFIG
TRAINING_CONFIG = {
    "epochs" : 50,
    "batch_size" : 32,
    "learning_rate" : 0.001,
    "weight_decay" : 1e-5,
    "devcie" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_every" : 10
}

# EVALUATOIN CONFIG
EVALUATION_CONFIG = {
    "contamination_rate" : 0.1,
    "percentile" : 95
}

# PATH
PATHS = {
    "outputs" : {
        "models" : "outputs/models/",
        "logs" : "outputs/logs/",
        "figures" : "outputs/figures/"
    },
    "data" : {
        "raw" : "data/raw/",
        "processed" : "data/processed/"
    }
}