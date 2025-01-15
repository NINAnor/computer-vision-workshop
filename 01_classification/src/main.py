import logging

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch
from dataset import get_train_val_dataloaders
from trainer import PetClassifier

if __name__ == "__main__":
    # dataset path
    data_dir = "/home/taheera.ahmed/code/computer-vision-workshop/01_classification/data/PetImages"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(
                "/home/taheera.ahmed/code/computer-vision-workshop/newfile.log",
                mode="w",
            ),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()

    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")

    train_loader, val_loader = get_train_val_dataloaders(
        logger, data_dir, batch_size=4, num_workers=4
    )


    # define callbacks
    early_stopping = EarlyStopping('val_loss') # monitor the validation loss
    model_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        filename='best_model',
    ) # save the best model with the lowest validation loss
    
    # train the model
    model = PetClassifier(num_classes=2)
    
    trainer = Trainer(
        max_epochs=10,
        accelerator="cpu", # TODO: Change to "gpu" for GPU training
        logger=True,
        log_every_n_steps=1,
        callbacks=[early_stopping, model_checkpoint]
    )
    trainer.fit(model, train_loader, val_loader)
