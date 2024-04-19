from shufflenet.shufflenet import ShuffleNet
from shufflenet.shufflenetv2 import ShuffleNetV2
from train import *


def modify_classifier(model, num_classes):
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


def load_pretrained_shufflenetv2(num_new_classes):
    shuffnet = ShuffleNet(1, width_mult=0.5)
    try:
        shuffnet.load_state_dict(torch.load("shufflenet/jester_shufflenet_0.5x_G3_RGB_16_best.pth", map_location=torch.device('cpu')))
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
    dataset = GIFDataset()
    train_data, val_data = train_val_sklearn_split(dataset, test_size=0.2)
    # unfortunately, it seems like the model script was corrupted and state dicts do not match
    model = load_pretrained_shufflenetv2(8)
    train(model, train_data, val_data)

