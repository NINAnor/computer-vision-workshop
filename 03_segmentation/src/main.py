#!/usr/env/bin python3

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchmetrics
import torchvision.models.segmentation as models
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import MulticlassJaccardIndex

from dataset import get_data_loaders
from transforms import albumentations_transform, resize_transform

torch.backends.cudnn.benchmark = True


def get_deeplabv3_model(num_classes):
    model = models.deeplabv3_resnet50(weights="COCO_WITH_VOC_LABELS_V1")
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    return model


class SegmentationModel(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-4):
        super().__init__()
        self.model = get_deeplabv3_model(num_classes)
        self.lr = lr
        self.num_classes = num_classes

        # Metrics from torchmetrics
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=-1
        )
        self.train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.train_iou = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=-1
        )

        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=-1
        )
        self.val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro", ignore_index=-1
        )
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=-1)

    def forward(self, x):
        return self.model(x)["out"]

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = F.cross_entropy(outputs, masks, ignore_index=-1)

        pred = torch.argmax(outputs, dim=1)

        # Log metrics using torchmetrics
        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy(pred, masks))
        self.log("train_precision", self.train_precision(pred, masks))
        self.log("train_recall", self.train_recall(pred, masks))
        self.log("train_iou", self.train_iou(pred, masks))

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        val_loss = F.cross_entropy(outputs, masks, ignore_index=-1)

        pred = torch.argmax(outputs, dim=1)

        # Log validation metrics
        self.log("val_loss", val_loss)
        self.log("val_accuracy", self.val_accuracy(pred, masks))
        self.log("val_precision", self.val_precision(pred, masks))
        self.log("val_recall", self.val_recall(pred, masks))
        self.log("val_iou", self.val_iou(pred, masks))

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    train_loader, val_loader = get_data_loaders(
        cfg["IMG_DIR"],
        cfg["MASKS_DIR"],
        batch_size=cfg["BATCH_SIZE"],
        albumentations_transform=albumentations_transform,
        resize_transform=resize_transform,
        num_workers=cfg["NUM_WORKERS"],
    )

    num_classes = cfg["NUM_CLASSES"]
    model = SegmentationModel(num_classes=num_classes, lr=cfg["LR"])

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg["PATIENCE"],
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=cfg["NUM_EPOCHS"],
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
