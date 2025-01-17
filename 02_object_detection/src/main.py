from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        project="02_object_detection",
        data="coco8.yaml",
        epochs=100,
        imgsz=640,
        workers=8,
    )