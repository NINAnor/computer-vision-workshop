from pathlib import Path

import gradio as gr
import hydra
from omegaconf import DictConfig
from ultralytics import YOLO


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # extract paths from the config file
    model_path = cfg.MODEL_PATH
    dataset_path = Path(cfg.DATASET_PATH)

    test_images_path = dataset_path / "test" / "images"

    # load the trained YOLO model
    model = YOLO(model_path)

    def run_inference(image_path):
        """Run inference on the provided image path."""
        result = model.predict(image_path)
        return result[0].plot()

    def get_example_images():
        """Retrieve example images from the test directory."""
        example_images = list(test_images_path.glob("*.jpg"))[:10]
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

    demo.launch()


if __name__ == "__main__":
    main()
