import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ============================
# 超参数
# ============================
n_steps = 20         # 输入序列长度
hidden_size = 64     # RNN 隐藏层大小
num_epochs = 200     # 训练轮数
lr = 0.01            # 学习率

# ============================
# 数据生成
# ============================
x_np = np.linspace(0, 20 * np.pi, 1000)
y_np = np.sin(x_np)

# 将 sin 序列拆成多个样本，每个样本是 n_steps 个输入 + 1 个预测目标
def create_dataset(data, n_steps):
    X, Y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        Y.append(data[i+n_steps])
    return np.array(X), np.array(Y)

X, Y = create_dataset(y_np, n_steps)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [batch, seq_len, input_size=1]
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # [batch, output=1]

# ============================
# RNN 模型定义
# ============================
class RNNPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(RNNPredictor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # out: [batch, seq_len, hidden_size]
        last_hidden = out[:, -1, :]  # 取最后一个时间步
        return self.fc(last_hidden)  # 输出预测值

model = RNNPredictor(hidden_size=hidden_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ============================
# 模型训练
# ============================
for epoch in range(num_epochs):
    pred = model(X)
    loss = criterion(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# ============================
# 可视化预测
# ============================
with torch.no_grad():
    pred = model(X).squeeze().numpy()

plt.figure(figsize=(10, 4))
plt.plot(y_np[n_steps:], label="True")
plt.plot(pred, label="Predicted")
plt.title("RNN Predicting sin(x)")
plt.legend()
plt.grid(True)
plt.show()

# ============================
# 使用新数据测试模型
# ============================
# 生成新的 sin(x) 数据（和训练集不重复）
x_test_np = np.linspace(20 * np.pi, 24 * np.pi, 400)
y_test_np = np.sin(x_test_np)

# 创建测试样本
X_test, Y_test = create_dataset(y_test_np, n_steps)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(-1)

# 模型预测
with torch.no_grad():
    y_pred = model(X_test).squeeze().numpy()
    y_true = Y_test.squeeze().numpy()

# 可视化预测效果
plt.figure(figsize=(10, 4))
plt.plot(y_true, label="True (New Data)")
plt.plot(y_pred, label="Predicted (New Data)")
plt.title("RNN Prediction on New sin(x) Sequence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
