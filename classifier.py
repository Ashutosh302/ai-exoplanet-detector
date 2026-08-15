"""
src/classifier.py
PyTorch 1D-CNN Architecture and inference helper for exoplanet light curve classification.
"""

import torch
import torch.nn as nn
import numpy as np


class ExoplanetCNN1D(nn.Module):
    """
    1D Convolutional Neural Network for classifying folded, binned light curve signals.
    """
    def __init__(self, input_length: int = 500, num_classes: int = 3):
        super(ExoplanetCNN1D, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 3
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # Calculate flattened dimension after 3 pooling layers: 500 -> 250 -> 125 -> 62
        flat_dim = (input_length // 8) * 64
        
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 1, input_length)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


CLASSES = ["Exoplanet Candidate", "Eclipsing Binary", "Noise / False Positive"]


def predict_lightcurve(model, folded_vector: np.ndarray, device: str = "cpu"):
    """
    Runs model inference on a single 1D light curve array.
    """
    model.eval()
    model.to(device)
    
    # Reshape input to (1, 1, sequence_length)
    tensor_input = torch.tensor(folded_vector).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_class_idx])
        
    return CLASSES[pred_class_idx], confidence, probabilities