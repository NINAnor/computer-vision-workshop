from pathlib import Path

import gradio as gr
from ultralytics import YOLO

# define the test images path
TEST_IMAGES_PATH = Path(
    "/home/taheera.ahmed/code/computer-vision-workshop/02_object_detection/data/aquarium_pretrain/test/images"
)

# load the YOLO model
model = YOLO(
    "/home/taheera.ahmed/code/computer-vision-workshop/02_object_detection/train_logs/train2/weights/best.pt"
)


def run_inference(image_path):
    result = model.predict(image_path)
    return result[0].plot()

def get_example_images():
    example_images = list(TEST_IMAGES_PATH.glob("*.jpg"))[:10] 
    return [str(img) for img in example_images]


# gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# YOLO Model Inference")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload an Image for Inference", type="filepath"
            )
        with gr.Column():
            output_image = gr.Image(label="Inference Result")

    infer_button = gr.Button("Run Inference")

    gr.Markdown("## Example Images")
    gr.Examples(
        examples=get_example_images(),
        inputs=image_input,
        label="Test Images",
        examples_per_page=8,
    )

    infer_button.click(fn=run_inference, inputs=image_input, outputs=output_image)

if __name__ == "__main__":
    demo.launch()
