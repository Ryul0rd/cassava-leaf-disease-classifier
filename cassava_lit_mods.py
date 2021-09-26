import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import wandb
import numpy as np
import pandas as pd


class CassavaDataMod(pl.LightningDataModule):
    def __init__(self, batch_size=32, val_size=0.2, balance_train=False, balance_val=False):
        super().__init__()
        self.data_dir = 'data/'
        self.batch_size = batch_size
        self.val_size = val_size
        self.balance_train = balance_train
        self.balance_val = balance_val
        self.augs = A.Compose([
            A.RandomResizedCrop(384, 384),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

    def aug_transform(self, image):
        return self.augs(image=np.array(image))['image'] # Dumb glue to accomodate albumentations

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.data = ImageFolder(self.data_dir, transform=self.aug_transform)

        # making weighted random samplers to deal with class balance
        df = pd.read_csv('data/train.csv')
        class_counts = df['label'].value_counts()
        class_weights = class_counts.median()/class_counts
        weights = []
        for class_id in range(5):
            weights.extend([class_weights[class_id] for i in range(class_counts[class_id])])
        weights = np.array(weights)

        val_count = int(len(self.data) * self.val_size)
        train_count = len(self.data) - val_count
        is_val = np.concatenate((np.ones(val_count, dtype=bool), np.zeros(train_count, dtype=bool)))
        np.random.shuffle(is_val)
        is_train = np.logical_not(is_val)

        train_weights = is_train * (weights if self.balance_train else 1)
        val_weights = is_val * (weights if self.balance_val else 1)
        if self.balance_train:
            self.train_sampler = WeightedRandomSampler(train_weights, int(len(self.data)*(1-self.val_size)), replacement=True)
        else:
            self.train_sampler = WeightedRandomSampler(train_weights, int(len(self.data)*(1-self.val_size)), replacement=False)
        if self.balance_val:
            self.val_sampler = WeightedRandomSampler(val_weights, int(len(self.data)*self.val_size), replacement=True)
        else:
            self.val_sampler = WeightedRandomSampler(val_weights, int(len(self.data)*self.val_size), replacement=False)

    def train_dataloader(self):
        return DataLoader(self.data, sampler=self.train_sampler, batch_size=self.batch_size, num_workers=3, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.data, sampler=self.val_sampler, batch_size=self.batch_size, num_workers=3, pin_memory=True)

    def test_dataloader(self):
        print('No test dataloader available. Using val dataloader instead.')
        return self.val_dataloader()


class CassavaLitMod(pl.LightningModule):
    def __init__(self, transformer_size='tiny', learning_rate=3e-4, hidden_size=0, weight_decay=1e-5):
        super().__init__()
        self.lr = learning_rate
        self.wd = weight_decay
        self.num_classes = 5
        self.accuracy = Accuracy()

        self.model = timm.create_model(f'vit_{transformer_size}_patch16_384', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        if hidden_size == 0:
            self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)
        else:
            self.model.head = nn.Sequential(
                nn.Linear(self.model.head.in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_classes),
            )

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + epoch))
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val/loss'
        }

    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        preds = self.model(x)
        loss = nn.functional.cross_entropy(preds, y_true)
        self.log('train/loss', loss, on_step=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        preds = self.model(x)
        loss = nn.functional.cross_entropy(preds, y_true)
        acc = self.accuracy(torch.argmax(preds, dim=1), y_true)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/accuracy', acc, prog_bar=True)
        return preds, y_true

    def validation_epoch_end(self, validation_step_outputs):
        preds, y_true = [], []
        for output in validation_step_outputs:
            preds.append(output[0])
            y_true.append(output[1])
        flattened_logits = torch.flatten(torch.cat(preds))
        self.logger.experiment.log(dict(
            val_logits = flattened_logits.cpu(),
            global_step = self.global_step,
        ))

        if self.current_epoch+1 != self.logger.experiment.config.epochs:
            return
        preds, y_true = torch.cat(preds).argmax(dim=1), torch.cat(y_true)
        preds, y_true = np.array(preds.cpu()), np.array(y_true.cpu())
        self.logger.experiment.log(dict(
            confmat = wandb.plot.confusion_matrix(preds=preds, y_true=y_true, title='Confusion Matrix', class_names=['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']),
            global_step = self.global_step,
        ))