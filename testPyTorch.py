import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 生成模拟数据：y = sin(x)
x = torch.unsqueeze(torch.linspace(-2 * torch.pi, 2 * torch.pi, 100), dim=1)
y = torch.sin(x)

# 数据预处理, 归一化
x = (x - x.mean()) / x.std()

# 简单的全连接神经网络：输入1维，隐藏层10个节点，输出1维
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Softmax(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNet()
criterion = nn.MSELoss()
lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=lr)

final_loss = 0

# 训练网络
for epoch in range(1000):
    pred = model(x)
    loss = criterion(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # if (epoch + 1) % 20 == 0:
    #     print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    final_loss = loss.item()

print(f'lr: {lr}')
print(f'Final Loss: {final_loss:.4f}')

# 绘图结果
plt.figure()
plt.plot(x.numpy(), y.numpy(), label='True')
plt.plot(x.numpy(), model(x).detach().numpy(), label='Predicted')
plt.legend()
plt.title("Simple Test: Fit sin(x)")
plt.show()