from lightning.pytorch import LightningModule
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class PetClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super(PetClassifier, self).__init__()
        self.save_hyperparameters()

        # define the model
        self.model = nn.Sequential(
            nn.Conv2d(
                3, 16, kernel_size=7, stride=2, padding=3
            ),  # Larger kernel, reduce initial spatial size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial size further
            nn.Flatten(),
            nn.Linear(16 * 56 * 56, 64),  # Fewer units in the fully connected layer
            nn.ReLU(),
            nn.Linear(64, num_classes),  # Output layer
        )

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
        loss = F.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
