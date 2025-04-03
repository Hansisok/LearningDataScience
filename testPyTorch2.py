import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ========================
# 归一化工具类
# ========================
class Normalizer:
    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse(self, norm_data):
        return norm_data * self.std + self.mean

# ========================
# 数据生成 & 归一化
# ========================
x_raw = torch.unsqueeze(torch.linspace(-2 * torch.pi, 2 * torch.pi, 1000), dim=1)
y_raw = torch.sin(x_raw)

x_norm = Normalizer()
y_norm = Normalizer()

# 先 fit 再 transform
x_norm.fit(x_raw)
y_norm.fit(y_raw)

x = x_norm.transform(x_raw)
y = y_norm.transform(y_raw)

# ========================
# 改进神经网络结构
# ========================
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ImprovedNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========================
# 训练模型
# ========================
for epoch in range(1000):
    pred = model(x)
    loss = criterion(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# ========================
# 可视化预测（反归一化）
# ========================
with torch.no_grad():
    pred = model(x)
    pred_true = y_norm.inverse(pred)

plt.figure(figsize=(8, 4))
plt.plot(x_raw.numpy(), y_raw.numpy(), label='True sin(x)', linewidth=1.5)
plt.plot(x_raw.numpy(), pred_true.numpy(), label='Predicted', linewidth=1.5)
plt.title("Neural Network Fit for sin(x) with Normalization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
