import torch

import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from model_3dconv import GIFClassifier
from dataset import GIFDataset, train_val_sklearn_split
import time
import os
import numpy as np
num_classes = 10

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'



def train_one_epoch(model: nn.Module, train_data: GIFDataset, val_data: GIFDataset, batch_size: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, plot: bool, epoch: int):
    model.train()

    if plot:
        train_losses = []
        val_accs = []
    subset = Subset(train_data, np.random.choice(len(train_data), size=len(train_data) // 10 + 1, replace=False))

    train_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=8)
    running_loss = 0
    with tqdm(train_dataloader, desc=f"Epoch {epoch+1}") as t:
        for i, sample in enumerate(t):
            inputs = sample["gif"].to(device)
            labels = torch.argmax(torch.stack(sample["target"]), dim=0).to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Update running loss and loss list
            running_loss += loss.item()
            if plot:
                train_losses.append(loss.item())

            # Update tqdm bar to show the current loss and running average loss
            t.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f} Avg: {running_loss / (i + 1):.4f}")
            t.refresh()  # to show immediately the update

    # Accuracy check at the end of each epoch
    train_acc = estimate_accuracy(model, train_data, "Training Accuracy")
    val_acc = estimate_accuracy(model, val_data, "Validation Accuracy")
    print(f"End of Epoch {epoch} - Training Accuracy: {train_acc}, Validation Accuracy: {val_acc}")

    # if plot:
    #     plt.figure()
    #     plt.plot(train_losses)
    #     plt.xlabel("Iterations")
    #     plt.ylabel("Training Loss")
    #     plt.title(f"Training Loss in Epoch {epoch}")
    #     plt.savefig(f'training_plots/loss_epoch{epoch}.png')

    #     plt.figure()
    #     plt.plot([val_acc] * len(train_losses))  # Plotting validation accuracy constant over the number of iterations
    #     plt.xlabel("Iterations")
    #     plt.ylabel("Validation Accuracy")
    #     plt.title(f"Validation Accuracy in Epoch {epoch}")
    #     plt.savefig(f'training_plots/accuracy_epoch{epoch}.png')

@torch.no_grad()
def get_accuracy(model: nn.Module, data: GIFDataset) -> float:
    
    model.eval()
    dataloader = DataLoader(data, batch_size=32)

    count = 0
    total = 0
    for sample in tqdm(dataloader, desc="test"):
        inputs = sample["gif"].to(device)
        labels = torch.argmax(torch.stack(sample["target"]), dim=0).to(device)

        logits = model(inputs)
        preds = torch.argmax(logits, 1)
        count += torch.sum(preds == labels)
        total += inputs.size(0)

    return count / total
@torch.no_grad()
def estimate_accuracy(model: nn.Module, data: GIFDataset, name: str) -> float:
    model.eval()
    
    # Efficient subset selection
    t1 = time.time()
    indices = np.random.choice(len(data), size=len(data) // 20 + 1, replace=False)
    subset = Subset(data, indices)
    
    # DataLoader with multiple workers
    dataloader = DataLoader(subset, batch_size=32, num_workers=8, pin_memory=True)
    t2 = time.time()
    print(f"DataLoader setup time: {t2 - t1}")
    count = 0
    total = 0
    
    # Main loop
    for sample in tqdm(dataloader, desc=name):
        inputs = sample["gif"].to(device)
        labels = torch.argmax(torch.stack(sample["target"]), dim=0).to(device)
        
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)
        
        count += torch.sum(preds == labels).item()  # Ensure count is incremented correctly
        total += inputs.size(0)
    
    accuracy = count / total if total > 0 else 0  # Handle division by zero
    return accuracy

def train(model: nn.Module, train_data: GIFDataset, val_data: GIFDataset, batch_size: int = 32, num_epochs: int = 100, lr: float = 0.001, weight_decay: int = 0.0, plot: bool = True, eval_every: int = 50, save_every: int = 10):
    print(f'Training on {device}')

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    
    if plot and not os.path.exists('training_plots'):
        os.makedirs('training_plots')
    train_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f"Training epoch {epoch}.")
        train_one_epoch(model, train_data, val_data, batch_size=batch_size, optimizer=optimizer, criterion=criterion, plot=plot, epoch=epoch)

        if epoch % save_every == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pth")


def main():
    dataset = GIFDataset()
    train_data, val_data = train_val_sklearn_split(dataset, test_size=0.2)
    model = GIFClassifier(num_classes).to(device)
    train(model, train_data, val_data)


if __name__ == "__main__":
    main()