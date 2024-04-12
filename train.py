
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model_2_1dconv import GIFClassifier
from dataset import GIFDataset

from typing import Tuple

def train_one_epoch(model: nn.Module,
                    train_data: GIFDataset,
                    val_data: GIFDataset,
                    batch_size: int,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    plot: bool,
                    plot_every: int):
    
    model.train()

    num_iters = []
    train_losses = []
    train_accs = []
    val_accs = []

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for i, sample in tqdm(enumerate(train_dataloader), desc='Epoch'):
        inputs = sample['gif']
        labels = sample['target']
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if plot and i % plot_every == 0:
            num_iters.append(i)
            train_loss = loss.item()
            train_acc = get_accuracy(model, train_data)
            val_acc = get_accuracy(model, val_data)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f'Iteration {i} training loss: {train_loss}, training accuracy: {train_acc}, validation accuracy: {val_acc}')

            model.train()

    plt.figure()
    plt.plot(num_iters, train_losses)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs. Number of Iterations')
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
def get_accuracy(model: nn.Module,
                 data: GIFDataset) -> Tuple[float, float]:
    
    model.eval()
    dataloader = DataLoader(data, batch_size=256)
    
    count = 0
    total = 0
    for sample in dataloader:
        inputs = sample['gif']
        labels = sample['target']
        
        logits = model(inputs)
        preds = torch.argmax(logits, 1)
        count += torch.sum(preds == labels)
        total += inputs.size(0)

    return count / total


def train(model: nn.Module,
          train_data: GIFDataset,
          val_data: GIFDataset,
          batch_size: int=64,
          num_epochs: int=10,
          lr: float=0.001,
          weight_decay: int=0.0,
          plot: bool=True,
          plot_every: int=50):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    for i in range(num_epochs):
        print(f'Training epoch {i + 1}.')
        train_one_epoch(model,
                        train_data,
                        val_data,
                        batch_size=batch_size,
                        optimizer=optimizer,
                        criterion=criterion,
                        plot=plot,
                        plot_every=plot_every)
    

def main():
    model = GIFClassifier()
    train_data = GIFDataset()
    val_data = GIFDataset()
    train(model, train_data, val_data)

    torch.save(model.state_dict(), '2_1d_model.pth')


if __name__ == '__main__':
    main()
