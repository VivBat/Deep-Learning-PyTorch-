import torch
from torch import nn
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split
from helper_functions import plot_predictions, plot_decision_boundary

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create data
X_blob, y_blob = make_blobs(n_samples=1000,
                               n_features=2,
                               centers=4,
                               cluster_std=1.5,
                               random_state=42)


print(f"X_blob.shape: {X_blob.shape}, y_blob.shape: {y_blob.shape}")

# visualise the data
plt.scatter(x=X_blob[:,0],
            y=X_blob[:,1],
            c=y_blob,
            cmap=plt.cm.RdYlBu)
# plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_blob,
                                                    y_blob,
                                                    test_size=0.2,
                                                    random_state=42)

print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

# turning the data into tensors
X_train = torch.from_numpy(X_train).type(torch.float).to(device)
y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
X_test = torch.from_numpy(X_test).type(torch.float).to(device)
y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)


# defining an accuracy function
def accuracy_fn(y_true, y_preds):
    equalities = torch.eq(y_true, y_preds)
    no_of_equalities = sum(equalities).item()
    acc = (no_of_equalities / len(y_preds))*100
    return acc

# Create the model
class MulticlassClassification(nn.Module):
    def __init__(self, input_features, output_features, neurons=8):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=output_features)
        )

    def forward(self, x):
        return self.layer_stack(x)


model = MulticlassClassification(input_features=2,
                                 output_features=4,
                                 neurons=8).to(device)


# y_untrained_preds = model(X_train)
# print(y_untrained_preds[:10])
# print(y_train[:10])

#loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.01)

# training
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100

for epoch in range(epochs):
    model.train()

    y_logits = model(X_train).squeeze()
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_preds=y_preds)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # testing
    model.eval()
    with torch.inference_mode():
        y_test_logits = model(X_test).squeeze()
        y_test_preds = torch.softmax(y_test_logits, dim=1).argmax(dim=1)
        loss_test = loss_fn(y_test_logits, y_test)
        acc_test = accuracy_fn(y_true=y_test, y_preds=y_test_preds)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | loss: {loss} | accuracy: {acc}% | Test loss: {loss_test} | Tests accuracy: {acc_test}%")


# making some prediction
model.eval()
with torch.inference_mode():
    y_logits = model(X_test)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

print(f"Predictions for first 10: {y_preds[:10]}")
print(f"actual for first 10: {y_test[:10]}")

plt.figure(figsize=(10,7))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()