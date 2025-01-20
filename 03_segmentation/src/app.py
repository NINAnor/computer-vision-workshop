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
model = load_model("/home/benjamin.cretois/Code/computer-vision-workshop/lightning_logs/version_1/checkpoints/epoch=27-step=476.ckpt", num_classes=8)

# Define preprocessing function
def preprocess_image(image):
    preprocess = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    return input_tensor

# Define a colormap for the 8 categories
def apply_colormap(predicted_mask):
    # Define colors for each class (8 classes)
    colors = [
        (0, 0, 0),       # Black - "innendørs"
        (255, 0, 0),     # Red - "parkering/sykkelstativ"
        (0, 255, 0),     # Green - "asfalt/betong"
        (0, 0, 255),     # Blue - "gummifelt/kunstgress"
        (255, 255, 0),   # Yellow - "sand/stein"
        (255, 0, 255),   # Magenta - "gress"
        (0, 255, 255),   # Cyan - "trær"
    ]

    # Convert class indices to RGB colors
    colored_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        colored_mask[predicted_mask == class_id] = color

    return Image.fromarray(colored_mask)

# Prediction function for Gradio
def predict(image):
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)["out"]

    # Get predicted mask (class-wise)
    predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Apply colormap to predicted mask
    colored_mask = apply_colormap(predicted_mask)

    # Resize the mask to match the input image size
    colored_mask = colored_mask.resize(image.size, Image.NEAREST)

    return colored_mask


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