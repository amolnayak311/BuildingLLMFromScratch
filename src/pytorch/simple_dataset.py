import torch
from torch.utils.data import Dataset, DataLoader

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])


class ToyDataset(Dataset):

        def __init__(self, X, y):
            self.features = X
            self.labels = y

        def __len__(self):
            return self.features.shape[0]

        def __getitem__(self, idx):
            one_x = self.features[idx]
            one_y = self.labels[idx]
            return one_x, one_y


train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)



train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0, drop_last=True)

if __name__ == "__main__":
    print(len(train_ds))
    for idx, (x, y) in enumerate(train_dl):
        print(f"Batch {idx + 1}:", x, y)