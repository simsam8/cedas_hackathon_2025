import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model import Regressor
from utils import prep_data

DATA_FOLDER = Path("./cedas2025_material/data")
data_path = DATA_FOLDER / "chargecurves_train.parquet"

train_data, val_data = prep_data(data_path)

batch_size = 512
NUM_WORKERS = int(os.cpu_count() / 2)


transform = transforms.ToTensor()

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, num_workers=NUM_WORKERS
)

val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=batch_size, num_workers=NUM_WORKERS
)

model = Regressor()

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    devices="auto",
    logger=TensorBoardLogger(save_dir="logs/"),
    callbacks=EarlyStopping("val_loss", patience=7),
)

trainer.fit(model, train_loader, val_loader)
