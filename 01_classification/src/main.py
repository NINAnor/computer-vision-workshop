#!/usr/env/bin python3

import logging

import torch
from dataset import get_train_val_dataloaders
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from trainer import PetClassifier
import hydra

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    # dataset path
    data_dir = cfg.DATA_PATH
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(
                "main.log",
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
        logger, data_dir, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS
    )

    # define callbacks
    # monitor the validation loss
    early_stopping = EarlyStopping(
        "val_loss", patience=2
    )  
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        filename="best_model",
    )  # save the best model with the lowest validation loss
    custom_logger = TensorBoardLogger(
        save_dir="01_classification",  # set your custom log directory here
    )

    # train the model
    model = PetClassifier(num_classes=cfg.NUM_CLASSES)
    logger.info(model)

    trainer = Trainer(
        max_epochs=cfg.MAX_EPOCH,
        accelerator=cfg.ACCELERATOR,  # TODO: Change to "gpu" for GPU training
        log_every_n_steps=1,
        callbacks=[early_stopping, model_checkpoint],
        logger=custom_logger,
    )
    trainer.fit(model, train_loader, val_loader)
    logger.info("Training completed.")


if __name__ == "__main__":
    main()

