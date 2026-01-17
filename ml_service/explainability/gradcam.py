import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model, image_tensor, target_class):
    target_layers = [model.cnn.features.denseblock4]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=[ClassifierOutputTarget(target_class)]
    )[0]

    img = image_tensor[0].permute(1,2,0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    cam_img = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_img
