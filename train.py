import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model_3dconv import GIFClassifier
from dataset import GIFDataset, train_val_sklearn_split
import time
import os

num_classes = 10

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Training on {device}')


def train_one_epoch(model: nn.Module, train_data: GIFDataset, val_data: GIFDataset, batch_size: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, eval_every: int, plot: bool, epoch: int):

    model.train()

    if plot:
        num_iters = []
        train_losses = []
        train_accs = []
        val_accs = []

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for i, sample in tqdm(enumerate(train_dataloader), desc="Epoch"):
        inputs = sample["gif"].to(device)
        labels = torch.argmax(torch.stack(sample["target"]), dim=0).to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % eval_every == 0:
            train_loss = loss.item()
            train_acc = get_accuracy(model, train_data)
            val_acc = get_accuracy(model, val_data)

            print(f"Iteration {i} training loss: {train_loss}, training accuracy: {train_acc}, validation accuracy: {val_acc}")

            if plot:
                num_iters.append(i)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)

            model.train()

    if plot:
        plt.figure()
        plt.plot(num_iters, train_losses)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Loss")
        plt.title(f"Loss vs. Number of Iterations in Epoch {epoch}")
        plt.savefig(f'training_plots/loss_epoch{epoch}')

        plt.figure()
        plt.plot(num_iters, train_accs)
        plt.plot(num_iters, val_accs)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs. Number of Iterations in Epoch {epoch}")
        plt.legend(["Train", "Validation"])
        plt.savefig(f'training_plots/accuracy_epoch{epoch}')


@torch.no_grad()
def get_accuracy(model: nn.Module, data: GIFDataset) -> float:

    model.eval()
    dataloader = DataLoader(data, batch_size=128)

    count = 0
    total = 0
    for sample in dataloader:
        inputs = sample["gif"].to(device)
        labels = torch.argmax(torch.stack(sample["target"]), dim=0).to(device)

        logits = model(inputs)
        preds = torch.argmax(logits, 1)
        count += torch.sum(preds == labels)
        total += inputs.size(0)

    return count / total


def train(model: nn.Module, train_data: GIFDataset, val_data: GIFDataset, batch_size: int = 64, num_epochs: int = 100, lr: float = 0.001, weight_decay: int = 0.0, plot: bool = True, eval_every: int = 50, save_every: int = 10):

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    
    if plot and not os.path.exists('training_plots'):
        os.makedirs('training_plots')

    for epoch in range(num_epochs):
        print(f"Training epoch {epoch}.")
        train_one_epoch(model, train_data, val_data, batch_size=batch_size, optimizer=optimizer, criterion=criterion, eval_every=eval_every, plot=plot, epoch=epoch)

        if epoch % save_every == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pth")


def main():
    dataset = GIFDataset()
    train_data, val_data = train_val_sklearn_split(dataset, test_size=0.2)
    model = GIFClassifier(num_classes).to(device)
    train(model, train_data, val_data)


if __name__ == "__main__":
    main()
