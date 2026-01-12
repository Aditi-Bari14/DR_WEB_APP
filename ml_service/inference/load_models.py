# inference/load_models.py
from inference.models import CNN_GCN_Proto
import torch

MODEL_PATH = "models/cnn_gcn_protopnet_BEST4.pth"

def load_model(device="cpu"):
    """
    Load the trained multimodal DR model for inference.
    Returns the PyTorch model ready for prediction.
    """
    print("Loading multimodal DR model...")

    # Load checkpoint (dict containing model_state_dict, optimizer_state_dict, etc.)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Initialize model architecture
    num_tab_features = 6  # Replace with your number of tabular features
    num_classes = 5       # Number of DR classes
    model = CNN_GCN_Proto(num_tab_features=num_tab_features, num_classes=num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Important: set model to evaluation mode

    print("Model loaded successfully")
    return model
