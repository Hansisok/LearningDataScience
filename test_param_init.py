'''
测试不同初始化方式对训练结果的影响。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 简单全连接网络，支持不同初始化方式
class SimpleNet(nn.Module):
    def __init__(self, init_method):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

        if init_method == 'zero':
            nn.init.constant_(self.fc1.weight, 0)
            nn.init.constant_(self.fc2.weight, 0)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        elif init_method == 'he':
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 合成数据
torch.manual_seed(42)
X = torch.randn(1000, 10)
true_weights = torch.randn(10, 1)
y = X @ true_weights + torch.randn(1000, 1) * 0.1

# 训练函数
def train_model(init_method):
    model = SimpleNet(init_method)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    losses = []
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# 对比三种初始化方式
loss_zero = train_model('zero')
loss_xavier = train_model('xavier')
loss_he = train_model('he')

# 画图
plt.figure(figsize=(10, 6))
plt.plot(loss_zero, label='Zero Init')
plt.plot(loss_xavier, label='Xavier Init')
plt.plot(loss_he, label='He Init')
plt.title('Training Loss vs Initialization Method')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()
