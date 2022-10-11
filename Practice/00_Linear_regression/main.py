import time

import torch
from torch import nn
import matplotlib.pyplot as plt

# Check PyTorch version
pytorch_version = torch.__version__
print(f"PyTorch version: {pytorch_version}")

# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Creating some sample data to use for linear regression
# y = w*x + b
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y
X = torch.arange(start, end, step).unsqueeze(dim=1)  # our features
y = weight * X + bias  # our  labels, we'll figure out weight and bias using the model, which should turn out to be 0.7 and 0.3 respectively

print(f"X[:10]: {X[:10]}, y[:10]: {y[:10]}")

# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# Plotting the data
def plot_prediction(train_data = X_train,
                    train_labels = y_train,
                    test_data = X_test,
                    test_labels = y_test,
                    predictions = None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="r", s=20, label="Train data")
    plt.scatter(test_data, test_labels, c="b", s=20, label="Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c = "y", s = 3, label="Predictions")
    plt.legend()
    plt.show()


# PyTorch Linear regression model

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # # Setting random values for start
        # self.weight = nn.Parameter(torch.randn(1,
        #                                        requires_grad=True, # this parameter, ie weight needs grad descent to be optimised
        #                                        dtype=torch.float))
        # self.bias = nn.Parameter(torch.randn(1,
        #                                       requires_grad=True,
        #                                       dtype=torch.float))
        #

        # Or

        # Using nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, # input of size 1 (X has only 1 feature, so only 1 w)
                                      out_features=1) # output of size 1 too

    # In case weight and bias are set manually
    # Forward method to define computation
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.weights * x + self.bias

    # when using torch's linear_layer
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


# set the manual seed
torch.manual_seed(69)

model_0 = LinearRegressionModel()
print(f"Model is: {model_0}")
print(f"Model's state_dict: {model_0.state_dict()}")

y_pred_before_training = model_0(X_test)
plot_prediction(predictions=y_pred_before_training.detach().numpy())
# Check the current model device
print("Model is currently on: ")
print(next(model_0.parameters()).device)

# setting the model to use the target device
model_0.to(device)
print("Model is now on: ")
print(next(model_0.parameters()).device)

# putting data on the target device too
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Training
# t1 = time.time()

loss_fn = nn.L1Loss()   # L1 loss is used as the loss function
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)  # Stochastic grad descent

torch.manual_seed(69)

epochs = 200

for epoch in range(epochs):
    model_0.train() # setting the model for training

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 ==0:
        print(f"Epoch: {epoch},| Loss: {loss},| Test loss = {test_loss}")

# t2 = time.time()
# print(f"time taken: {t2-t1}")

print(model_0.state_dict())

# Making predictions

model_0.eval()

with torch.inference_mode():
    y_preds = model_0(X_test) # now prediction will be made with the newly calculated  weight and bias after the training

# plotting after training
plot_prediction(predictions=y_preds.to("cpu").numpy())