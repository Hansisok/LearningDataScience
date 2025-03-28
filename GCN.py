import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import KarateClub
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载数据
dataset = KarateClub(transform=NormalizeFeatures())
data = dataset[0]  # 仅有一个图

# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# 设置训练/测试掩码
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[:4] = True  # 少量节点用于训练
data.test_mask = ~data.train_mask

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试函数
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    return acc

# 训练过程
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

# 可视化嵌入
@torch.no_grad()
def visualize():
    model.eval()
    out = model(data.x, data.edge_index)
    z = TSNE(n_components=2).fit_transform(out.cpu())
    plt.figure(figsize=(8,6))
    plt.scatter(z[:,0], z[:,1], c=data.y.cpu(), cmap="Set2", s=100)
    plt.title("Node Embedding Visualization (GCN)")
    plt.show()

visualize()
