import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
import wandb
from torchvision.transforms import v2
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(batch_size):
    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # download MNIST training dataset
    # Load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


class SequentialCIFARModel(nn.Module):
    def __init__(self, input_shape, num_classes, learning_rate=3e-4,
                 fc_layer_size=None, dropout=None):
        super(SequentialCIFARModel, self).__init__()

        # Define the layers sequentially
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )

        # Calculate the size of the output from the convolutional layers
        n_sizes = self._get_output_shape(input_shape)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(n_sizes, fc_layer_size),
            nn.ReLU(),
            nn.Linear(fc_layer_size, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1)
        )

        # log hyperparameters
        self.learning_rate = learning_rate

    def _get_output_shape(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    elif optimizer == 'adamw':
        optimizer = optim.AdamW(network.parameters(),
                                lr=learning_rate)
    elif optimizer == 'nadam':
        optimizer = optim.NAdam(network.parameters(),
                                lr=learning_rate)

    elif optimizer == 'rprop':
        optimizer = optim.Rprop(network.parameters(),
                                lr=learning_rate)

    return optimizer


def train_epoch(network, loader, optimizer):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        loss = F.cross_entropy(network(data), target)
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, entity='deep-learning-lab-msc'):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = SequentialCIFARModel(input_shape=(3, 32, 32), num_classes=10,
                                       fc_layer_size=config.fc_layer_size, dropout=config.dropout).to(device)
        optimizer = build_optimizer(network, config.optimizer, config.lr)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})




