import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from trainer import PetClassifier

# Function to load the model
def load_model(model_path):
    # Load the checkpoint
    checkpoint = torch.load(model_path, weights_only=True)
    
    state_dict = checkpoint["state_dict"]
    
    model = PetClassifier()
    model.load_state_dict(state_dict)
    model.eval() 
    
    return model
# Define the inference function
def predict(image, model_path):
    # Load the model
    model = load_model(model_path)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).squeeze()

    # Map predictions to labels
    classes = ["Cat", "Dog"]
    predictions = {classes[i]: float(probabilities[i]) for i in range(len(classes))}

    return predictions

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="filepath", label="Upload an Image"),
        gr.Textbox(lines=1, label="Model Path"),
    ],
    outputs=gr.Label(num_top_classes=2, label="Predictions"),
    title="Pet Classifier",
    description="Upload an image of a cat or dog and provide the file path to the trained model. The app will predict whether the image is of a cat or a dog.",
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()
