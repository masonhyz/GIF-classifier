
import torch

from model_2_1dconv import GIFClassifier
from torch.utils.data import Dataset
from train import train

num_classes = 10

class GIFDataset(Dataset):
    """Custom GIF dataset for testing model."""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        labels = self.labels[idx]
        sample = {'gif': data, 'target': labels}
        return sample
    
def main():
    train_data = torch.randn(2, 50, 3, 500, 500)
    train_labels = torch.randint(0, num_classes, size=(2,))
    train_dataset = GIFDataset(train_data, train_labels)

    val_data = torch.randn(1, 50, 3, 500, 500)
    val_labels = torch.randint(0, num_classes, size=(1,))
    val_dataset = GIFDataset(val_data, val_labels)
    
    model = GIFClassifier(num_classes)
    train(model, train_dataset, val_dataset, 1, num_epochs=2, plot_every=1)

if __name__ == '__main__':
    main()
