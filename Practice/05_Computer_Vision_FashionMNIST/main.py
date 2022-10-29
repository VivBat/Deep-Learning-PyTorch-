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
from tqdm.auto import tqdm
import numpy as np

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from pathlib import Path


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 3

train_data = datasets.FashionMNIST(root='data',
                                   train=True,
                                   transform=ToTensor(),
                                   target_transform=None,
                                   download=True)

test_data = datasets.FashionMNIST(root='data',
                                  train=False,
                                  transform=ToTensor(),
                                  target_transform=None,
                                  download=True)

print(f"Shape of training data is: {train_data.data.shape}")  # so 60000 images, each of size 28x28
print(f"Shape of testing data is: {test_data.data.shape}")  # so 10000 images, each of size 28x28

label_classes = train_data.classes

# # just checking the first image in the dataset
img0, label0 = train_data.data[0], train_data.targets[0].item()
# print(f"First image is: {img0}")
# print(f"First label is: {label0}")

# plt.imshow(img0, cmap='gray')
# plt.title(label_classes[label0])
# plt.show()

# Visualising 16 random images from the training data
# plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows * cols + 1):
#     random_idx = random.randint(0, len(train_data))
#     img, label = train_data.data[random_idx], train_data.targets[random_idx].item()
#     plt.subplot(rows, cols, i)
#     plt.imshow(img, cmap='gray')
#     plt.title(label_classes[label])
#     plt.axis(False)
# plt.show()


#  Dataloader
# Dataloader to convert the dataset into iterables of a given batch size
train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

train_dataloader0_data, train_dataloader0_label = (next(iter(train_dataloader)))  # the first image

print(f"Number of batches: {len(train_dataloader)}")
print(f"Shape of one of the iterable datasets (dataloader): {train_dataloader0_data.shape}")



# The model
class CNN_FashionMNIST_V01(nn.Module):
    """
       Replicates the TinyVGG architecture from (https://poloclub.github.io/cnn-explainer/)
    """

    def __init__(self, input_shape, output_shape, neurons):
        super().__init__()
        self.conv2d_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=neurons,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU()
        )

        self.conv2d_block2 = nn.Sequential(
            nn.Conv2d(in_channels=neurons,
                      out_channels=neurons,
                      kernel_size=3,
                      stride=1,
                      padding=0
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=1,
                         padding=0)
        )

        self.classification_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=neurons * 23 * 23,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv2d_block1(x)
        # print(f"Output of block1: {x.shape}")
        x = self.conv2d_block2(x)
        # print(f"Output of block2: {x.shape}")
        x = self.classification_layer(x)
        # print(f"Output of classification layer: {x.shape}")
        return x


def train_step(model: nn.Module,
               data: DataLoader,
               loss_function: nn.CrossEntropyLoss,
               optimizer: torch.optim,
               accuracy_function):
    """
    Performs the training step for a given model
    :param model: an nn.Module model which is to be trained
    :param data: the training data
    :param loss_function: loss function to be used for training
    :param optimizer: optimizer to be used for training
    :param accuracy_function: function that should be used to calculate accuracy
    :return: Training loss, Training accuracy
    """

    # setting the model to training mode
    model.train()

    train_loss = 0  # to collect the training loss for the entire train dataset
    train_accuracy = 0  # to collect the training accuracy for the entire train dataset

    for batch, (X, y) in enumerate(data):  # going over each batch once at a time

        # sending the data to device
        X = X.to(device)
        y = y.to(device)

        # Forward prop
        y_logits = model(X)

        # loss for just the current batch
        loss = loss_function(y_logits, y)

        # adding the loss of the current batch to the loss of the entire dataset
        train_loss += loss

        # converting the logits to predictions
        y_preds = torch.softmax(y_logits, dim=0).argmax(dim=1)

        # calculating the accuracy from y_preds
        train_accuracy += accuracy_function(y, y_preds)

        # setting the grad to zero
        optimizer.zero_grad()

        # back prop
        loss.backward()

        # updating the parameters with the new weights and biases
        optimizer.step()

        if batch % 400 == 0:
            print(f"Reached upto batch no: {batch}")

    # since the loss and accuracy for all the batches are added up now, dividing by the size
    train_loss /= len(data)
    train_accuracy /= len(data)

    return train_loss, train_accuracy


def test_step(model: nn.Module,
              data: DataLoader,
              loss_function: nn.CrossEntropyLoss,
              accuracy_function):
    """
      Performs the testing step for a given model
      :param model: an nn.Module model which is to be tested on the testing data
      :param data: the testing data
      :param loss_function: loss function to be used for testing
      :param accuracy_function: function that should be used to calculate accuracy
      :return: Testing loss, Testing accuracy
      """

    # setting the model to eval mode and with inference mode
    model.eval()

    test_loss = 0  # to collect test loss for the entire test dataset
    test_accuracy = 0  # to collect test accuracy for the entire test dataset

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data):
            # sending the data to device
            X = X.to(device)
            y = y.to(device)

            # forward prop
            y_logits = model(X)

            # calculating loss and adding it to test_loss
            test_loss += loss_function(y_logits, y)

            # prediction from logits
            y_preds = torch.softmax(y_logits, dim=0).argmax(dim=1)

            # calculating the test accuracy
            test_accuracy += accuracy_function(y, y_preds)

        test_loss /= len(data)
        test_accuracy /= len(data)

    return test_loss, test_accuracy


def model_eval(model: nn.Module,
               data: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               accuracy_function
               ):
    loss, acc = 0, 0
    # setting the model to eval mode
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data):
            X = X.to(device)
            y = y.to(device)
            # Forward prop
            y_logits = model(X)

            # loss
            loss += loss_function(y_logits, y)

            # converting to predictions
            y_preds = torch.softmax(y_logits, dim=0).argmax(dim=1)

            # accuracy
            acc += accuracy_function(y, y_preds)

        loss /= len(data)
        acc /= len(data)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_accuracy": acc}


# instantiating the model
model = CNN_FashionMNIST_V01(input_shape=1,
                             output_shape=len(label_classes),
                             neurons=10).to(device)

# model(train_dataloader0_data.to(device))

# setting up the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

train_loss_hist = []
test_loss_hist = []
# training and testing the model
for epoch in tqdm(range(EPOCHS)):
    print(f"----Epoch: {epoch} ------")
    train_loss, train_acc = train_step(model=model,
                                       data=train_dataloader,
                                       loss_function=loss_fn,
                                       optimizer=optimizer,
                                       accuracy_function=accuracy_fn)

    test_loss, test_acc = test_step(model=model,
                                    data=test_dataloader,
                                    loss_function=loss_fn,
                                    accuracy_function=accuracy_fn)

    print(
        f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}% | Test loss: {test_loss:.4f} | "
        f"Test acc: {test_acc:.2f}%"
    )

model_result = model_eval(model,
                          test_dataloader,
                          loss_function=loss_fn,
                          accuracy_function=accuracy_fn)

print(model_result)


def make_predictions(model: nn.Module,
                     data: torch.utils.data.DataLoader):
    """
    Makes predictions for all the images in the test data
    :param model: The trained model
    :param data: Data for which predictions are to be made
    :return: A tensor containing predictions for the entire dataset
    """
    model.eval()

    preds = []
    with torch.inference_mode():
        for X, y in data:
            X = X.to(device)

            y_logits = model(X)

            y_pred = torch.softmax(y_logits, dim=0).argmax(dim=1)

            preds.append(y_pred)

    predictions = torch.cat(preds)

    return predictions


predictions_on_test_data = make_predictions(model,
                                            data=test_dataloader)

# print(f"preds comparison---: {predictions_on_test_data == test_data.targets.to(device)}")

rows, cols = 4, 4
plt.figure(figsize=(9,9))

for i in range(1, rows*cols+1):
    random_idx = random.randint(0,len(predictions_on_test_data))
    img, label, pred_label = test_data.data[random_idx], test_data.targets[random_idx], predictions_on_test_data[random_idx]

    plt.subplot(rows, cols, i)
    plt.imshow(img, cmap='gray')

    title_color = 'g' if label == pred_label else 'r'
    plt.title(label_classes[label] + " | " + label_classes[pred_label], c=title_color, fontsize=10)

    plt.axis(False)
plt.show()

# Plotting the confusion matrix
conf_mat = ConfusionMatrix(num_classes=len(label_classes))

conf_mat_tensor = conf_mat(preds=predictions_on_test_data.to("cpu"),
                           target=test_data.targets)

print(f"Confusion matrix: {conf_mat_tensor}")
# using mlxtend to visualise the confusion matrix

fig, ax = plot_confusion_matrix(conf_mat=conf_mat_tensor.numpy(),
                                class_names=label_classes,
                                figsize=(10,7))

plt.show()

# Saving the model
MODEL_PATH = Path("/home/vivek/Documents/FEL/Machine Learning/PyTorch/Practice/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "CNN_FashionMNIST_V01.pth"
MODEL_SAVE_PATH =MODEL_PATH / MODEL_NAME

print(f"Saving model to : {MODEL_PATH}")
torch.save(obj=model.state_dict(),  # Saving only the learned weights and biases of the trained model
           f=MODEL_SAVE_PATH)


