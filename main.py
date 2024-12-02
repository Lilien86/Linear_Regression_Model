import torch
from torch import nn
from pprint import pprint
from pathlib import Path
import matplotlib.pyplot as plt

print(f"PyTorch version : {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device use : {device}")




# Initialise data
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
print(X[:10]), print(y[:10])



# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
len(X_train), len(y_train), len(X_test), len(y_test)



# Visualizer
def plot_predictions(train_data=X_train, train_labels=y_train, 
                     test_data=X_test, test_labels=y_test, predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Données d'entraînement")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Données de test")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Prédictions")
    plt.legend(prop={"size": 14})
    plt.xlabel("Données (X)", fontsize=14)
    plt.ylabel("Valeurs (y)", fontsize=14)
    plt.title("Graphique des Prédictions", fontsize=16)
    plt.show()

plot_predictions()

# Create model Class and displace to GPU
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1.to(device)
next(model_1.parameters()).device



# Create loss fucntion and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)
# model_1 = model_1.state_dict()



# Train loop
epochs = 1000

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    
    model_1.train()
    y_pred = model_1(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)
    
    if (epoch % 100 ==0):
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")



# Print results
print("The model learned the following values for weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")



# Test prediction model
model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)
print(y_preds)
plot_predictions(predictions=y_preds.cpu())



# Store model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)
