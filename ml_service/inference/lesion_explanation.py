import numpy as np

def generate_lesion_text(predicted_class, cam_path=None):
    """
    Rule-based lesion explanation from DR stage
    """

    explanations = {
        0: {
            "stage": "No DR",
            "lesions": [
                "No visible microaneurysms",
                "No hemorrhages detected",
                "No exudates observed"
            ]
        },
        1: {
            "stage": "Mild DR",
            "lesions": [
                "Possible presence of microaneurysms",
                "No significant hemorrhages",
                "Minimal retinal damage"
            ]
        },
        2: {
            "stage": "Moderate DR",
            "lesions": [
                "Microaneurysms likely present",
                "Possible hemorrhages",
                "Hard exudates may be present"
            ]
        },
        3: {
            "stage": "Severe DR",
            "lesions": [
                "Multiple hemorrhages likely",
                "Cotton wool spots possible",
                "Extensive retinal damage"
            ]
        },
        4: {
            "stage": "Proliferative DR",
            "lesions": [
                "High probability of neovascularization",
                "Severe hemorrhages detected",
                "High risk of vision loss"
            ]
        }
    }

    return explanations.get(predicted_class, {})
