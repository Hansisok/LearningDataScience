import torch
import torch.nn as nn

# 定义 PINN 网络
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, t):
        return self.net(t)

# 微分方程: du/dt = -u, u(0)=1
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练数据（t点）
t = torch.linspace(0, 5, 100).view(-1, 1).requires_grad_(True)

# 训练循环
for epoch in range(5000):
    u = model(t)
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    residual = du_dt + u

    # 初始条件损失
    u0 = model(torch.tensor([[0.0]]))
    loss = torch.mean(residual**2) + (u0 - 1)**2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4e}")