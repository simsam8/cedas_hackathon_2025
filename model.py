import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError


class SimlpeLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4, hidden_size=40, num_layers=8, batch_first=True
        )
        self.fc = nn.Linear(40, 40)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        x = hidden.squeeze()[-1]  # get last layer of lstm
        x = F.relu(self.fc(x))
        return x


class RegressorLSTM(pl.LightningModule):
    def __init__(self, lr=0.001) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SimlpeLSTM()
        self.loss_fn = nn.MSELoss()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        mae = self.train_mae(pred, y)
        self.log("train_mae", mae)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss)
        mae = self.val_mae(pred, y)
        self.log("val_mae", mae)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Regressor(pl.LightningModule):
    def __init__(self, lr=0.001) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = Net()
        self.loss_fn = nn.MSELoss()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        mae = self.train_mae(pred, y)
        self.log("train_mae", mae)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss)
        mae = self.val_mae(pred, y)
        self.log("val_mae", mae)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
