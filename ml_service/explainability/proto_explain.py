import torch
import numpy as np

def proto_explanation(model, image_tensor):
    with torch.no_grad():
        img_feat = model.cnn(image_tensor)
        proj = model.img_proj(img_feat)
        proto_sim, _ = model.proto(proj)

    proto_sim = proto_sim[0].cpu().numpy()
    top = np.argsort(-proto_sim)[:5]

    return {
        "top_prototypes": top.tolist(),
        "similarity_scores": proto_sim[top].tolist()
    }
