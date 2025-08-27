import torch

from src.pytorch.simple_dataset import train_dl, X_train, test_dl
from src.pytorch.simple_nn import NeuralNetwork

device = torch.device("mps")
torch.manual_seed(123)
model = NeuralNetwork(num_input=2, num_output=2).to(device)

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.5)

print("Num parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
num_epochs =3


for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_dl):
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_dl):03d}"
              f" | Train Loss: {loss:.2f}")

#model = model.to("cpu")
model.eval()
with torch.no_grad():
    outputs = model(X_train.to(device))
print(outputs)

torch.set_printoptions(sci_mode=False)
probas = torch.softmax(outputs, dim=1)
print(probas)

print(torch.argmax(probas, dim = 1))

print(model)

def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0
    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

print(compute_accuracy(model, train_dl))
print(compute_accuracy(model, test_dl))

torch.save(model.state_dict(), "model.pt")

dup_model = NeuralNetwork(num_input=2, num_output=2)
dup_model.load_state_dict(torch.load("model.pt", weights_only=True))

print(dup_model)