import pytorch_lightning as pl
import torch
from torch import nn


class LinearModel(pl.LightningModule):
    def __init__(
        self,
        input_dim=20,
        layers=5,
        hidden_dim=128,
        output_classes=3,
        dropout_p=0.5,
        class_weights=None,
    ):
        super().__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes
        self.dropout = nn.Dropout(dropout_p)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, output_classes),
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, inputs, labels=None):
        output = self.model(inputs)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        output = self.forward(x)
        loss = self.criterion(output, y)
        y_hat = torch.argmax(output, dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        self.log("train_accuracy", accuracy)
        self.log("train_loss", loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().long()
        output = self.forward(x)
        loss = self.criterion(output.squeeze(), y)
        y_hat = torch.argmax(output, dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        self.log("valid_accuracy", accuracy)
        self.log("valid_loss", loss.detach())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.8, patience=25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
            },
        }
