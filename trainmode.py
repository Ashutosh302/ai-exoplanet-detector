"""
train_model.py
Trains the 1D-CNN using simulated and labeled light curves.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.classifier import ExoplanetCNN1D


def generate_synthetic_dataset(num_samples_per_class: int = 1500, length: int = 500):
    """
    Creates a synthetic dataset of 3 classes:
    Class 0: Exoplanet (U-shaped flat bottom dip)
    Class 1: Eclipsing Binary (V-shaped deep dip, possible secondary eclipse)
    Class 2: Noise / Instrumental trends (Pure white + red noise)
    """
    X = []
    y = []
    phase = np.linspace(-0.5, 0.5, length)
    
    for _ in range(num_samples_per_class):
        noise = np.random.normal(0, np.random.uniform(0.0005, 0.003), length)
        
        # 1. Exoplanet (U-shaped transit)
        depth_planet = np.random.uniform(0.002, 0.02)
        dur_planet = np.random.uniform(0.02, 0.08)
        flux_planet = np.zeros(length)
        in_transit = np.abs(phase) < (dur_planet / 2.0)
        flux_planet[in_transit] = -depth_planet
        X.append((flux_planet + noise).astype(np.float32))
        y.append(0)
        
        # 2. Eclipsing Binary (V-shaped primary + secondary dip)
        depth_eb1 = np.random.uniform(0.05, 0.25)
        depth_eb2 = depth_eb1 * np.random.uniform(0.2, 0.6)
        dur_eb = np.random.uniform(0.05, 0.15)
        # V-shape profile
        flux_eb = np.zeros(length)
        v_mask1 = np.abs(phase) < (dur_eb / 2.0)
        flux_eb[v_mask1] = -depth_eb1 * (1 - np.abs(phase[v_mask1]) / (dur_eb / 2.0))
        # Secondary eclipse at phase ~ 0.5 or -0.5
        v_mask2 = np.abs(np.abs(phase) - 0.5) < (dur_eb / 2.0)
        flux_eb[v_mask2] = -depth_eb2 * (1 - np.abs(np.abs(phase[v_mask2]) - 0.5) / (dur_eb / 2.0))
        X.append((flux_eb + noise).astype(np.float32))
        y.append(1)
        
        # 3. Pure Noise / Stellar Variability (Sinusoidal or Random Walk)
        variability = np.random.uniform(0.001, 0.005) * np.sin(2 * np.pi * phase * np.random.uniform(1, 4))
        X.append((variability + noise).astype(np.float32))
        y.append(2)
        
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]


def train():
    os.makedirs("./models", exist_ok=True)
    print("Generating training dataset...")
    X, y = generate_synthetic_dataset(num_samples_per_class=1200)
    
    # Reshape for 1D CNN: (N, 1, 500)
    X_tensor = torch.tensor(X).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Split Train/Val (80/20)
    n_train = int(0.8 * len(X))
    train_dataset = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
    val_dataset = TensorDataset(X_tensor[n_train:], y_tensor[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExoplanetCNN1D().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 15
    print(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
        val_acc = val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Train Acc: {train_acc*100:.1f}% - Val Acc: {val_acc*100:.1f}%")
        
    model_path = "./models/exoplanet_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")


if __name__ == "__main__":
    train()