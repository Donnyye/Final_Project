import torch
from torch import nn
import torch.nn.functional as F


# --- Model Definitions ---
class LogisticModel(nn.Module):
    """
    Logistic Regression Model
    Structure:
    Z = Xw + b
    y = softmax(Z)
    """

    def __init__(self, input_size=(32, 32), n_classes=10, in_channels=3):
        super(LogisticModel, self).__init__()
        self.flatten = nn.Flatten()
        flatten_size = in_channels * input_size[0] * input_size[1]
        self.sequential = nn.Sequential(
            nn.Linear(flatten_size, n_classes),
            nn.Softmax(dim=1)
        )

        self.sequential.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # xavier init
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, X):
        X = self.flatten(X)
        y = self.sequential(X)

        return y


class FCN(nn.Module):
    """
    ML model for Fully-connected network
    Structure:
    h1 = Xw
    h2 = h1w
    y = h2w
    """

    def __init__(self, input_size=(32, 32), n_classes=10, in_channels=3):
        super(FCN, self).__init__()
        self.flatten = nn.Flatten()
        flatten_size = in_channels * input_size[0] * input_size[1]
        self.fcnn = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

        self.fcnn.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # xavier init
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, X):
        X = self.flatten(X)
        y = self.fcnn(X)

        return y


class ConvNet(nn.Module):
    """
    Convolution neural network model vgg-16
    """

    def __init__(self, input_size=(32, 32), n_classes=10, in_channels=3):
        super(ConvNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),  # 64 * 32 * 32
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64 * 32 * 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 * 16 * 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128 * 16 * 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 * 16 * 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 * 8 * 8
        )

        self.flatten = nn.Flatten()
        flatten_size = 128 * 8 * 8
        self.fcnn = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=1)
        )

        self.convs.apply(self.init_weights)
        self.fcnn.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # xavier init
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, X):
        X = self.convs(X)
        X = self.flatten(X)
        y = self.fcnn(X)

        return y


class Baseline(nn.Module):
    """
    Baseline model with 1/k accuracy, predict everything class 0
    """

    def __init__(self, device, input_size=(32, 32), n_classes=10, in_channels=3):
        super(Baseline, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.device = device

    def forward(self, X):
        X_shape = X.shape[0]
        y = torch.zeros(size=(X_shape, self.n_classes)).to(self.device)
        y[:, 0] += torch.ones(size=(X_shape,)).to(self.device)

        return y
