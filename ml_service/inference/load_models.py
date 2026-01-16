# inference/load_models.py
from inference.models import CNN_GCN_Proto
import torch

MODEL_PATH = "models/cnn_gcn_protopnet_BEST4.pth"

def load_model(device="cpu"):
    """
    Load the trained multimodal DR model for inference.
    """
    print("Loading multimodal DR model...")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = CNN_GCN_Proto(
        num_tab_features=6,
        num_classes=5
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Model loaded successfully")
    return model
