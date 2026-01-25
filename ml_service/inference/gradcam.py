import torch
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, class_idx):
        """
        image_tensor: [1, 3, 224, 224]
        class_idx: int
        """

        self.model.zero_grad()

        # 🔹 Forward
        output = self.model(image_tensor)

        # 🔹 DenseNet output is feature tensor → fake logits
        score = output[:, class_idx].sum()

        # 🔹 Backward
        score.backward()

        # 🔹 Global average pooling
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        cam = cv2.resize(cam, (224, 224))

        return cam


# def save_gradcam(cam, original_image_path, output_path):
#     img = cv2.imread(original_image_path)
#     img = cv2.resize(img, (224, 224))

#     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#     overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

#     cv2.imwrite(output_path, overlay)