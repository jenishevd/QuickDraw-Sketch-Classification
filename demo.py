"""
Interactive Gradio demo — draw a sketch, get a prediction.
"""

import torch
import numpy as np
from PIL import Image
import gradio as gr
from train import CLASSES
from model import SketchCNN

CHECKPOINT = "sketch_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SketchCNN(num_classes=len(CLASSES)).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()


def predict(input_image):
    if input_image is None:
        return {}

    # Gradio sketchpad returns a dict — grab the merged composite image
    if isinstance(input_image, dict):
        composite = input_image.get("composite")
        input_image = composite if composite is not None else input_image.get("image")

    if input_image is None:
        return {}

    try:
        arr = np.asarray(input_image)
        valid_array = (
            arr.size > 0
            and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] in (1, 3, 4)))
            and np.issubdtype(arr.dtype, np.number)
            and np.isfinite(arr).all()
        )
    except (TypeError, ValueError):
        return {}

    if not valid_array:
        return {}

    # Convert to grayscale using RGB channels (NOT alpha).
    # The sketchpad has a white background (RGB=255) with black strokes (RGB=0).
    # Alpha is 255 everywhere on a white-background canvas, so it carries no info.
    if arr.ndim == 3:
        gray = arr[..., :3].mean(axis=-1)   # average R, G, B → white=255, ink=0
    else:
        gray = arr.astype(np.float32)

    # Scale to uint8 if needed (Gradio sometimes returns float 0–1)
    if gray.max() <= 1.0:
        gray = gray * 255.0

    # Require at least 0.1% of the canvas to contain ink (with a two-pixel
    # floor).  This rejects an empty canvas and isolated dots while allowing
    # thin but intentional sketches; the check happens before downsampling so
    # a tiny mark cannot turn into a seemingly meaningful 28x28 feature.
    if gray.mean() > 127:
        ink_mask = gray < 245
    else:
        ink_mask = gray > 10
    min_ink_pixels = max(2, int(np.ceil(gray.size * 0.001)))
    if np.count_nonzero(ink_mask) < min_ink_pixels:
        return {"Draw something recognizable": 1.0}

    bitmap = np.array(Image.fromarray(gray.astype(np.uint8)).convert("L").resize((28, 28)))

    # QuickDraw: bright = ink. Invert if background is bright.
    if bitmap.mean() > 127:
        bitmap = 255 - bitmap

    tensor = torch.tensor(bitmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()

    top1_confidence = float(probs.max())
    # With 12 classes, an uninformative softmax is 1/12 (about 0.083).
    # Requiring 0.42—roughly five times that baseline—avoids presenting a
    # weak preference as a prediction. Temperature calibration on validation
    # data would make this cutoff more reliable, but is out of scope here.
    if top1_confidence < 0.42:
        return {"Not confident enough — try drawing more clearly": 1.0}

    top3 = probs.argsort()[-3:][::-1]
    return {CLASSES[i]: float(probs[i]) for i in top3}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(type="numpy", image_mode="RGBA"),
    outputs=gr.Label(num_top_classes=3),
    title="QuickDraw Sketch Classifier",
    description="Draw one of: " + ", ".join(CLASSES),
)

demo.launch()
