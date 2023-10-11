import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassConfusionMatrix

from torchvision.transforms import v2
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download the CIFAR-10 dataset
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    # dataloaders
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)


class CIFARLitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes,
                 wandb_logger=None, trainer=None, learning_rate=3e-4):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # model architecture
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)

        n_sizes = self._get_output_shape(input_shape)

        # linear layers for classifier head
        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)  # Task of the students
        self.fc4 = nn.Linear(128, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=10)

    def _get_output_shape(self, shape):
        """returns the size of the output tensor from the conv layers"""

        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._feature_extractor(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # computations
    def _feature_extractor(self, x):
        """extract features from the conv blocks"""
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x

    def forward(self, x):
        """produce final model output"""
        x = self._feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

    # train loop
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # metric
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    # test loop
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    # optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class ResNetLightning(pl.LightningModule):
    def __init__(self, resnet_model=None,
                 learning_rate=0.001,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.resnet = resnet_model
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=10)

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # metric
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer
