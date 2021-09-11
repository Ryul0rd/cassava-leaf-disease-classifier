import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

class CassavaLeafDataMod(pl.LightningDataModule):
    def __init__(self, batch_size=32, train_percent=0.8):
        super().__init__()
        self.data_dir = 'data/'
        self.batch_size = batch_size
        self.train_percent = train_percent
        self.transform = T.Compose([
            T.ToTensor(),
            T.RandomResizedCrop(size=(28, 28)),
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        data = ImageFolder(self.data_dir, transform=self.transform)
        train_count = int(len(data) * self.train_percent)
        val_count = len(data) - train_count
        self.train_data, self.val_data = random_split(data, [train_count, val_count])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        print('No test dataloader available. Using val dataloader instead.')
        return self.val_dataloader()

class CassavaLeafLitMod(pl.LightningModule):
    def __init__(self, lr=1e-3, hidden_size=32, wd=1e-5):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.accuracy = Accuracy()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*28*28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_loss'
        }

    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        y_pred = self.net(x)
        loss = nn.functional.cross_entropy(y_pred, y_true)
        self.log('train_loss', loss, prog_bar=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        y_pred = self.net(x)
        loss = nn.functional.cross_entropy(y_pred, y_true)
        acc = self.accuracy(torch.argmax(y_pred, dim=1), y_true)
        self.log('val_loss', loss, prog_bar=True)
        self.log('accuracy', acc, prog_bar=True)