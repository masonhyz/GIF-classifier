
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import GIFClassifier

from typing import Tuple

def train_one_epoch(model: nn.Module,
                    train_data: DataLoader,
                    val_data: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    plot_every: int=50):
    
    model.train()

    num_iters = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for i, (data, label) in tqdm(enumerate(train_data), desc='Epoch'):
        logits = model(data)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % plot_every == 0:
            num_iters.append(i)
            train_loss, train_acc = evaluate(model, train_data)
            val_loss, val_acc = evaluate(model, val_data)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

    plt.figure()
    plt.plot(num_iters, train_losses)
    plt.plot(num_iters, val_losses)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs. Number of Iterations')
    plt.legend(['Train', 'Validation'])
    plt.show()

    plt.figure()
    plt.plot(num_iters, train_accs)
    plt.plot(num_iters, val_accs)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Iterations')
    plt.legend(['Train', 'Validation'])
    plt.show()


@torch.no_grad()
def evaluate(model: nn.Module,
             data: DataLoader) -> Tuple[float, float]:
    
    """TODO"""
    model.eval()
    loss = 1.0
    acc = 1.0

    model.train()
    return loss, acc


def train(model: nn.Module,
          train_data: DataLoader,
          val_data: DataLoader,
          num_epochs: int=10,
          lr: float=0.001):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for _ in range(num_epochs):
        train_one_epoch(model,
                        train_data,
                        val_data,
                        optimizer,
                        criterion)
    

def main():
    model = GIFClassifier()
    train_data = DataLoader()
    val_data = DataLoader()
    train(model, train_data, val_data)


if __name__ == '__main__':
    main()
