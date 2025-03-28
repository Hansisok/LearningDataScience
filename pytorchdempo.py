import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 MNIST 数据集
transform = transforms.ToTensor()
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# 定义一个简单神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 训练网络
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # ① 前向传播
        outputs = model(images)

        # ② 计算损失
        loss = loss_fn(outputs, labels)

        # ③ 清除旧梯度
        optimizer.zero_grad()

        # ④ 反向传播
        loss.backward()

        # ⑤ 更新参数
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

    print(f"→ Epoch {epoch+1} 结束，总Loss: {total_loss:.4f}")

# 测试准确率
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\n✅ 测试集准确率: {100 * correct / total:.2f}%")

# 可视化测试集中的预测
images, labels = next(iter(test_loader))
images, labels = images[:10].to(device), labels[:10].to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

# 可视化前 10 张图片及预测
plt.figure(figsize=(12, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
    plt.title(f"{preds[i].item()}")
    plt.axis('off')
plt.suptitle("前10个测试图像预测")
plt.show()
