import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh激活函数"""
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU激活函数"""
    return np.where(x > 0, x, alpha * x)

# 生成数据点
x = np.linspace(-10, 10, 400)

# 计算各激活函数的值
y_relu = relu(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_leaky_relu = leaky_relu(x)

# 绘制激活函数图像
plt.figure(figsize=(12, 8))

# ReLU
plt.subplot(2, 2, 1)
plt.plot(x, y_relu, label="ReLU Function")
plt.title("ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.grid(True)
plt.legend()

# Sigmoid
plt.subplot(2, 2, 2)
plt.plot(x, y_sigmoid, label="Sigmoid Function", color="orange")
plt.title("Sigmoid Activation Function")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid(True)
plt.legend()

# Tanh
plt.subplot(2, 2, 3)
plt.plot(x, y_tanh, label="Tanh Function", color="green")
plt.title("Tanh Activation Function")
plt.xlabel("x")
plt.ylabel("Tanh(x)")
plt.grid(True)
plt.legend()

# Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(x, y_leaky_relu, label="Leaky ReLU Function", color="red")
plt.title("Leaky ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("Leaky ReLU(x)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
