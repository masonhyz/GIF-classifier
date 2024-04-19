import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from shufflenet.shufflenetv2 import ShuffleNetV2
# from train import *

def modify_classifier(model, num_classes):
    in_features = model.classifier[1].in_features  # access in_features of the original classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),  # maintaining the same dropout rate
        nn.Linear(in_features, num_classes)  # new classifier with 'num_classes' outputs
    )
    return model


def load_pretrained_shufflenetv2(num_new_classes):
    # Initialize model with the original number of classes
    num_original_classes = 400  # This must match the original model's classifier output size
    groups = 3  # Example group number, adjust based on your architecture
    shuffnet = ShuffleNetV2()
    try:
        shuffnet.load_state_dict(torch.load("shufflenet/jester_mobilenetv2_1.0x_RGB_16_best.pth"))
        print("successfully loaded pretrained model")
    except Exception as e:
        print(e)
    # Modify the model for a new number of classes
    shuffnet = modify_classifier(shuffnet, num_new_classes)

    for param in shuffnet.parameters():
        param.requires_grad = False

    # Unfreeze the last classifier layer
    for param in shuffnet.classifier.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    # dataset = GIFDataset()
    # train_data, val_data = train_val_sklearn_split(dataset, test_size=0.2)
    model = load_pretrained_shufflenetv2(8)
    # train(model, train_data, val_data, start_epoch=start_epoch)


