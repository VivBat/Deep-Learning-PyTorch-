import random

import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from helper_functions import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm    # for progress bar visualisation

print(f"torchvision version: {torchvision.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 3

# Getting the dataset
# training data
train_data = datasets.FashionMNIST(root='data',
                                   train=True,
                                   transform=ToTensor(),
                                   target_transform=None,
                                   download=True)

# test data
test_data = datasets.FashionMNIST(root='data',
                                  train=False,
                                  transform=ToTensor(),
                                  target_transform=None,
                                  download=True)

image, label = train_data[0]  # first image
# print(train_data[0])
print(f"one image's shape: {image.shape}")
print(train_data, test_data)

print(f"Number of images: {len(train_data.data)}, Number of train labels: {len(train_data.targets)}, Number of test "
      f"images: {len(test_data.data)}, Number of test labels: {len(test_data.targets)}")

class_names = train_data.classes
class_names_idx = train_data.class_to_idx
print(f"Names of classes in the dataset: {class_names}")
print(f"Names of classes with their indices: {class_names_idx}")

# # visualizing 16 random images from the dataset
# fig = plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows * cols + 1):
#     # random_idx = torch.randint(0,len(train_data))
#     random_idx = random.randint(0, len(train_data))
#     image, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(image.squeeze(), cmap='gray')
#     plt.title(train_data.classes[label])
#     plt.axis(False)
# # plt.show()

# Turning the dataset into iterables with a certain batch size

train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True, )

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

print(f"Dataloaders: {train_dataloader},\n     {test_dataloader}")

print(f"Length of train dataloader: {len(train_dataloader)}, of batch size {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)}, of batch size {BATCH_SIZE}")


a_train_features_batch, a_train_labels_batch = next(iter(train_dataloader))  # gets the first batch of the dataloader
print(f"Shape (B x C x H x W) of a batch of the dataloader: Features: {a_train_features_batch.shape}, labels: {a_train_labels_batch.shape}")

# print(a_train_features_batch)
# visualizing the dataloader
random_idx = random.randint(0, len(a_train_features_batch))
img, label = a_train_features_batch[random_idx], a_train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap='gray')
plt.title(class_names[label])
plt.axis(False)
plt.show()


# understanding nn.flatten
flattener = nn.Flatten()

# for a single sample
x = a_train_features_batch[0]

flattened_sample = flattener(x)

# print(flattened_sample)
print(f"shape of flattened sample: {flattened_sample.shape}")  # changed the image from [color, height, width] to [color, height*width] -->into a one long feature vector
print(f"shape of the sample before flattening: {x.shape}")

# The model
class FashionMNISTmodelV0(nn.Module):
    def __init__(self, input_features, output_features, neurons):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=neurons),
            nn.Linear(in_features=neurons, out_features=output_features)
        )

    def forward(self, x):
        return self.layer_stack(x)


model_0 = FashionMNISTmodelV0(input_features=784,    # (28*28), one for every pixel of image 28x28
                              output_features=len(class_names),
                              neurons=10) #.to(device)

print(f"Model: {model_0}")

# setting up the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(),
                             lr=LEARNING_RATE)


# A function to measure time
def print_time_taken(start: float, end: float, device_used=None):
    time_taken = end - start
    print(f"Time taken on device {device_used} is: {time_taken}")
    return time_taken


# Training the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_start_time = timer()

epochs = EPOCHS


for epoch in tqdm(range(epochs)):

    train_loss = 0

    for batch, (X, y) in enumerate(train_dataloader):  # parameters (weights and biases) optimised updated with every batch
        model_0.train()

        # forward pass
        y_logits = model_0(X)
        # print(f"y_logits: {y_logits.shape}")

        # loss
        loss = loss_fn(y_logits, y)
        train_loss += loss

        # optimizer zero grad
        optimizer.zero_grad()

        # back propagation
        loss.backward()

        optimizer.step()

        if batch % 400 == 0:
            print(f"Reached upto batch no: {batch}")

    train_loss /= len(train_dataloader)

    # testing
    test_loss = 0
    test_acc = 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:   # after parameters optimised for a batch, testing done just to see the improvement
            test_logits = model_0(X_test)

            test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

            test_loss += loss_fn(test_logits, y_test)

            test_acc += accuracy_fn(y_true=y_test, y_pred=test_preds)

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"Epoch: {epoch} | Training loss: {train_loss} | Testing loss: {test_loss} | Test accuracy: {test_acc}")

train_end_time = timer()

print_time_taken(train_start_time, train_end_time, next(model_0.parameters()).device)

