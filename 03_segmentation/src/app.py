import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50

# Load the model
def load_model(checkpoint_path, num_classes):
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Remove the 'model.' prefix from the state_dict keys
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Load trained model
model = load_model("path/to/your/model_checkpoint.pth", num_classes=2)

# Define preprocessing function
def preprocess_image(image):
    preprocess = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    return input_tensor

# Prediction function for Gradio
def predict(image):
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)["out"]

    # Get predicted mask
    predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Convert mask to image format
    predicted_mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
    predicted_mask_image = predicted_mask_image.resize(image.size, Image.NEAREST)

    return predicted_mask_image

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Image Segmentation with DeepLabV3",
    description="Upload an image and get the corresponding segmented mask."
)

if __name__ == "__main__":
    iface.launch()