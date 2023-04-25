import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from dataset import Tox21Dataset


class Net(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch_geometric.nn.global_max_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# Load dataset
dataset = Tox21Dataset()
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Split dataset into train and test sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

# Initialize model
num_features = 2048
hidden_channels = 256
num_classes = 12
model = Net(num_features, hidden_channels, num_classes)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

# Train model
model.train()
for epoch in range(10):
    for data in loader:
        optimizer.zero_grad()
        x, edge_index, batch = data.x, data.edge_index, data.batch
        y = data.y[:, 0]
        out = model(x, edge_index, batch)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

# Evaluate model on test set
model.eval()
correct = 0
total = 0
for data in DataLoader(test_dataset, batch_size=64):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    y = data.y[:, 0]
    out = model(x, edge_index, batch)
    _, predicted = torch.max(out.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum().item()

print('Accuracy on test set: {:.2f}%'.format(100 * correct / total))
