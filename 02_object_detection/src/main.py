from ultralytics import YOLO
from pathlib import Path    

if __name__ == "__main__":

    data_dir = Path("02_object_detection/data/aquarium_pretrain")
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
    
    # TODO: check up for data augmentation and other hhy