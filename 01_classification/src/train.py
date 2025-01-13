import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.models import resnet18
from dataset import get_train_val_dataloaders

torch.set_float32_matmul_precision('medium')

class PetClassifier(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super(PetClassifier, self).__init__()
        self.save_hyperparameters()

        # use a pre-trained ResNet18 and modify for binary classification
        self.model = resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.binary_cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == "__main__":
    # dataset path
    data_dir = "/home/taheera.ahmed/code/computer-vision-workshop/01_classification/data/PetImages"

    # load train and validation DataLoaders
    train_loader, val_loader = get_train_val_dataloaders(data_dir, batch_size=32)

    # train the model
    model = PetClassifier(num_classes=2)
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        logger=True,
        log_every_n_steps=1, 
        num_sanity_val_steps=0, 
        fast_dev_run=True
    )
    trainer.fit(model, train_loader, val_loader)