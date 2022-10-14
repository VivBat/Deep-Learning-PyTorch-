import torch
from torch import nn
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd

# number of samples
n = 1000

#create circles
X, y = make_circles(n,
                    noise=0.01,
                    random_state=69)

print(f"X.shape: {X.shape}, y.shape: {y.shape}")
# print(f"first 5 samples: {X[:5], y[:5]}")

# make dataframe of circles
circles = pd.DataFrame({"X1": X[:,0],
                        "X2": X[:,1],
                        "label":y })
print(circles.head(10))

# print(circles.label.value_counts())  # How many values of each class is there?

# visualise the data
plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)
# plt.show()

# turning data into tensors
X = torch.from_numpy(X).type(torch.float)   # and also changed the dtype
y = torch.from_numpy(y).type(torch.float)

# print(X[:10], y[:10])

# splitting the data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=69)

print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}, y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape} ")


# make the code device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Available device: {device}")

### Building the model

# Just with linear activations
class BinaryClassification_v01(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer2(self.layer1(x))

# # another way to create the model is
# model = nn.Sequential(
#     nn.Linear(in_features=2, out_features=10),
#     nn.Linear(in_features=10, out_features=1)
# ).to(device)

# with ReLU activations
class BinaryClassification_v02(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x))))) # 2 relu layers


# creating an instance of the model and sending to cuda as well
model_0 = BinaryClassification_v01().to(device)  # with only linear

print(f"Model: {model_0}")

# lets pass the data through the model before it has been trained on anything
untrained_preds = model_0(X_test.to(device))
print(f"y_test[:10]: {y_test[:10]}")
print(f"untrained_pred[:10]: {untrained_preds[:10].squeeze()}")


# Accuracy method to determine accuracy of the model
def accuracy_fn(y_true, y_preds):
    equalities = torch.eq(y_preds,y_true)
    no_of_equalities = sum(equalities).item()
    acc = (no_of_equalities / len(y_preds))*100
    return acc


# lets setup the loss function and optimizer for training
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(),
                            lr=0.01)


# Train the model
torch.manual_seed(69)
torch.cuda.manual_seed(69)

# sending the data to the device (cuda)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epochs = 100

for epoch in range(epochs):

    model_0.train()  # setting the model to train mode

    # 1. Forward pass
    y_logits = model_0(X_train).squeeze()# the model returns the logits since loss function is BCE with logits
    y_preds = torch.round(torch.sigmoid(y_logits))  # logits -> probabilties -> turning to 0 or 1

    # 2. Calculating loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_preds)

    # 3. Setting grad to zero
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ## Testing
    model_0.eval()  # setting the model to evaluation mode

    with torch.inference_mode():
        # Forward pass
        test_logits = model_0(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        # Calculate loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_preds=test_preds)

    # printing
    if epoch % 10 ==0:
        print(f"Epoch: {epoch} | loss: {loss:.5f} | Accuracy: {acc:.2f} | Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}  ")


# BinaryClassification_v01 doesnt seem to do anything. The accuracy is 50%.
# Downloading helper functions to plot the decision boundary made by this model

if Path("helper_functions.py").is_file():  # if the file containing helper functions already exists
    print("The file already exists")
else:
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py") # getting the file from this url
    with open("helper_functions.py", "wb") as f:    # creating a file with the name
        f.write(request.content)                    # writing the contents of the request to the file


from helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()

model_1 = BinaryClassification_v02().to(device)
print(f"Model: {model_1}")

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_1.parameters(),  # Adam works wayyy better than just SGD
                            lr=0.1)

# Training the model

epochs = 1000

for epoch in range(epochs):

    model_1.train()

    y_logits = model_1(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_preds=y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_preds=test_preds)

    if epoch % 100 ==0:
        print(f"Epoch: {epoch} | loss: {loss:.5f} | accuracy: {acc:.2f} | Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}")


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
plt.show()