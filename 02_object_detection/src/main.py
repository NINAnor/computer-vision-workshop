from pathlib import Path

import hydra
from ultralytics import YOLO


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    data_dir = Path(cfg.DATA_PATH)
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data file not found: {data_yaml}")

    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    model.train(
        project="02_object_detection/train_logs",
        data=data_yaml,
        epochs=100,
        imgsz=640,
        workers=8,
    )

if __name__ == "__main__":
    main()
