import torch.nn

class NeuralNetwork(torch.nn.Module):

    def __init__(self, num_input, num_output):
        super(NeuralNetwork, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_input, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_output)
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    model = NeuralNetwork(50, 3)
    print(model)
    print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    torch.manual_seed(123)
    X = torch.rand((1, 50))
    out = model(X)
    print(out, out.shape)
    with torch.no_grad():
        out = model(X)
        print(out, torch.softmax(out, dim=1), torch.argmax(out, dim=1))
