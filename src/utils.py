"""
Helper
    - plot training loss.
    - save checkpoint dan load checkpoint.
"""

import torch 
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
# from config.CONFIG import PATHS

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Model checkpoint disimpan di {filename}")

def load_checkpoint(filename, device):
    if device.type == 'cuda':
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    print(f"Model checkpoint dimuat dari {filename}")
    return checkpoint

def plot_training_loss(train_losses, train_recon_losses = None, train_kld_losses = None, save_path = None):
    """
    Plot training loss
    """
    plt.figure(figsize=(12, 8))

    epochs = range(1, len(train_losses) + 1)

    # plot total loss
    plt.plot(epochs, train_losses, 'b-', linewidth = 2, label = 'Total Loss', alpha = 0.8)

    # plot recon loss
    if train_recon_losses is not None:
        plt.plot(epochs, train_recon_losses, 'g-', linewidth = 2, label = 'Reconstructionn Loss', alpha = 0.8)

    # plot kld loss
    if train_kld_losses is not None:
        plt.plot(epochs, train_kld_losses, 'r-', linewidth = 2, label = 'KL Divergence Loss', alpha = 0.8)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha = 0.3)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
        print(f"Training loss plot disimpan di {save_path}")

    # if save_path is None:
    #     save_path = os.path.join(PATHS["outputs"]["figures"], "training_loss.png")

    plt.show()