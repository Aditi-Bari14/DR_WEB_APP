import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import cosine

# Image preprocessing (same as model input)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_features(model, image_path, device):
    """
    Extract CNN features for an image
    """
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.cnn.features(img)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)

    return features.cpu().numpy()[0]


def find_most_similar_prototype(
    uploaded_image_path,
    prototypes_root,
    model,
    device
):
    """
    Compare uploaded image with all prototype images
    and return the most similar one
    """

    uploaded_feat = extract_features(
        model, uploaded_image_path, device
    )

    best_score = float("inf")
    best_image_path = None

    for class_name in os.listdir(prototypes_root):
        class_dir = os.path.join(prototypes_root, class_name)

        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            proto_path = os.path.join(class_dir, img_name)

            proto_feat = extract_features(
                model, proto_path, device
            )

            distance = cosine(uploaded_feat, proto_feat)

            if distance < best_score:
                best_score = distance
                best_image_path = proto_path

    return best_image_path