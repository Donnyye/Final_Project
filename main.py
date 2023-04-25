"""
Name: Hanzhe Ye
Student ID: 1671744
Class Section: CMPUT 466
GitHub Link for this project:

"""
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import LogisticModel, FCN, ConvNet, Baseline

train_iter = 0
writer = SummaryWriter()
# --- Data Loaders ---
def load_data(config):
    """
    Load Cifar-10 Dataset
    :param config: Data Configurations
    :return: train_dataloader, valid_dataloader, test_dataloader
    """
    transform_train = config["transform_train"]
    transform_test = config["transform_test"]
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # split into train and validation sets
    train_CFR = data.Subset(trainset, range(45000))
    valid_CFR = data.Subset(trainset, range(45000, 50000))
    train_dataloader = torch.utils.data.DataLoader(train_CFR, batch_size=config["batch_size"], shuffle=True,
                                                   num_workers=2)
    valid_dataloader = torch.utils.data.DataLoader(valid_CFR, batch_size=config["batch_size"], shuffle=True,
                                                   num_workers=2)

    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    return train_dataloader, valid_dataloader, test_dataloader

def train_loop(dataloader, model, loss_fn, device, config, model_name=""):
    """
    A one-step training iteration
    :return: train_losses
    """
    global train_iter
    size = len(dataloader.dataset)
    train_losses = []

    optimizer = torch.optim.SGD(params=model.parameters(), lr=config["lr"], momentum=config["momentum_SGD"],
                weight_decay=config["weight_decay"])

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        t = F.one_hot(y, num_classes=10).type(torch.float)
        X, t, y = X.to(device), t.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, t)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), config["grad_clip"])
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            train_losses.append(loss)

            train_iter += 1
            writer.add_scalar(model_name + "/Loss/Train", loss, train_iter)
            writer.flush()

    return train_losses


def valid_loop(dataloader, model, loss_fn, device):
    """
    A one-step validation
    :return: model.copy(), accuracy, valid_loss
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            t = F.one_hot(y, num_classes=10).type(torch.float)
            X, t, y = X.to(device), t.to(device), y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, t).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    valid_loss /= num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100.0 * accuracy):>0.1f}%, Avg loss: {valid_loss:>8f} \n")
    return copy(model), 100.0 * accuracy, valid_loss


def Train(model, loss_fn, train_dataloader, valid_dataloader, device, config, model_name=""):
    """
    Training Epochs
    :param model: Any machine learning model
    :param loss_fn: A loss criterion
    :param train_dataloader: Load Training Data
    :param valid_dataloader: Load Evaluation Data
    :param device: GPU device (e.g. cuda)
    :param config: Training Configurations
    :return: best_model, best_epoch, best_acc, best_loss, train_iter, train_losses, eval_losses, eval_accuracy
    """
    # train & validate
    global train_iter
    # total training iterations
    train_iter = 0
    # best parameters & training records
    best_acc = 0.0
    best_epoch = None
    best_model = None
    best_loss = None

    train_losses = []
    eval_losses = []
    eval_accuracy = []
    for t in range(config["num_epochs"]):
        print(f"Epoch {t + 1}\n-------------------------------")
        losses = train_loop(train_dataloader, model, loss_fn, device, config, model_name=model_name)
        train_losses += losses

        current_model, valid_acc, valid_loss = valid_loop(valid_dataloader, model, loss_fn, device)

        eval_losses.append(valid_loss)
        eval_accuracy.append(valid_acc)

        writer.add_scalar(model_name + "/Loss/Valid", valid_loss, t + 1)
        writer.add_scalar(model_name + "/Acc/Valid", valid_acc, t + 1)
        writer.flush()
        if valid_acc > best_acc:
            best_epoch = t + 1
            best_model = current_model
            best_loss = valid_loss
            best_acc = valid_acc

        del current_model
    print("Done!")

    return best_model, best_epoch, best_acc, best_loss, train_iter, train_losses, eval_losses, eval_accuracy


def Test(model, test_dataloader, device):
    """
    Model Testing
    :param model: Any machine learning model
    :param test_dataloader: Load Testing Data
    :param device: GPU device
    :return: test_predictions, true_labels, accuracy
    """
    test_predictions = []
    true_labels = []
    test_size = len(test_dataloader.dataset)
    correct = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            # prediction
            pred = model(X)
            test_predictions.append(pred.argmax(1))
            true_labels.append(y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    accuracy = correct / test_size
    print('Accuracy of the network on the 10000 test images: %.1f %%' % (100 * accuracy))
    return test_predictions, true_labels, 100.0 * accuracy


def plot_results(t_iter, train_losses, eval_losses, eval_acc, config, model_name):
    assert " " not in model_name, "Plotting error: Invalid File Name!"

    if not os.path.isdir("plots"):
        os.mkdir("plots")

    train_enum = np.linspace(1, t_iter, t_iter)
    epoch_enum = np.linspace(1, config["num_epochs"], config["num_epochs"])

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title(model_name + " training Losses")
    plt.xlabel("Iter")
    plt.ylabel("Loss")
    plt.plot(train_enum, train_losses)
    plt.subplot(222)
    plt.title(model_name + " eval Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_enum, eval_losses)
    plt.subplot(223)
    plt.title(model_name + " eval accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Acc (%)")
    plt.plot(epoch_enum, eval_acc)
    plt.savefig("plots/" + model_name + "_result.png")


def model_export(model, model_name):
    assert " " not in model_name, "Saving error: Invalid File Name!"

    if not os.path.isdir("weights"):
        os.mkdir("models")

    # Save a weight dict and a model
    torch.save(model.state_dict(), "weights/" + model_name + ".pt")


def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'lr': 0.01,
        'num_epochs': 50,
        'batch_size': 64,
        'grad_clip': 5.0,
        'num_classes': 10,
        'momentum_SGD': 0.9,
        'weight_decay': 1e-5,
        'transform_train': transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'transform_test': transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }
    train_loader, valid_loader, test_loader = load_data(config)

    print("Test result for Logistic model:")
    LoTestAcc = Test(Baseline(device).to(device), test_loader, device)
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'lr': 0.01,
        'num_epochs': 50,
        'batch_size': 64,
        'grad_clip': 5.0,
        'num_classes': 10,
        'momentum_SGD': 0.9,
        'weight_decay': 1e-5,
        'transform_train': transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'transform_test': transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    # Setting up models and losses
    Logit_model = LogisticModel(input_size=(32, 32), n_classes=10, in_channels=3).to(device)
    FCN_model = FCN(input_size=(32, 32), n_classes=10, in_channels=3).to(device)
    CNN_model = ConvNet(input_size=(32, 32), n_classes=10, in_channels=3).to(device)

    Baseline_model = Baseline(device=device, input_size=(32, 32), n_classes=10, in_channels=3).to(device)

    Logit_loss = nn.CrossEntropyLoss()
    FCN_loss = nn.MSELoss()
    CNN_loss = nn.CrossEntropyLoss()

    # Training process for all models
    train_loader, valid_loader, test_loader = load_data(config)
    print("Training CNN model:")
    best_CNN, CNNep, CNNacc, CNNloss, CNNTrainIter, CNNTrainloss, CNNEvalloss, CNNEvalAcc = \
        Train(CNN_model, CNN_loss, train_loader, valid_loader, device, config, model_name="CNN model")
    print()

    print("Training Logistic Model:")
    best_Logistic, Loep, Loacc, Loloss, LoTrainIter, LoTrainloss, LoEvalloss, LoEvalAcc = \
        Train(Logit_model, Logit_loss, train_loader, valid_loader, device, config, model_name="Logistic Model")
    print()

    print("Training FCN model:")
    best_FCN, FCNep, FCNacc, FCNloss, FCNTrainIter, FCNTrainloss, FCNEvalloss, FCNEvalAcc = \
        Train(FCN_model, FCN_loss, train_loader, valid_loader, device, config, model_name="FCN Model")
    print()

    writer.close()

    # Exporting Models
    print("Exporting Logistic Model:")
    model_export(best_Logistic, model_name="logistic_model")
    print("Logistic Model saved!")
    model_export(best_FCN, model_name="FCN_model")
    print("FCN Model saved!")
    model_export(best_CNN, model_name="CNN_model")
    print("CNN Model saved!")
    print()

    # Displaying validation Results
    print("Logistic Model validation information:")
    print("Best epoch: %d" % Loep)
    print("Best Accuracy: %.1f %%" % Loacc)
    print("Best Loss: %f" % Loloss)
    print()

    print("FCN Model validation information:")
    print("Best epoch: %d" % FCNep)
    print("Best Accuracy: %.1f %%" % FCNacc)
    print("Best Loss: %f" % FCNloss)
    print()

    print("CNN Model validation information:")
    print("Best epoch: %d" % CNNep)
    print("Best Accuracy: %.1f %%" % CNNacc)
    print("Best Loss: %f" % CNNloss)
    print()

    # Displaying Test Results
    print("Test result for Logistic model:")
    LoTestAcc = Test(best_Logistic, test_loader, device)
    print()

    print("Test result for FCN model:")
    FCNTestAcc = Test(best_FCN, test_loader, device)
    print()

    print("Test result for CNN model:")
    CNNTestAcc = Test(best_CNN, test_loader, device)
    print()

    print("Test result for Baseline model:")
    CNNTestAcc = Test(Baseline_model, test_loader, device)  # Baseline result
    print()

    # Plotting Results
    plot_results(LoTrainIter, LoTrainloss, LoEvalloss, LoEvalAcc, config, model_name="logistic_model")
    plot_results(FCNTrainIter, FCNTrainloss, FCNEvalloss, FCNEvalAcc, config, model_name="FCN_model")
    plot_results(CNNTrainIter, CNNTrainloss, CNNEvalloss, CNNEvalAcc, config, model_name="CNN_model")


if __name__ == '__main__':
    main()
