import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 定义数据预处理
transform = transforms.ToTensor()

# 2. 下载训练集和测试集
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 3. 构造 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. 基本信息检查
print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))

image, label = train_dataset[0]
print("Single image shape:", image.shape)   # 期望: [1, 28, 28]
print("Single label:", label)

# 5. 检查一个 batch
images, labels = next(iter(train_loader))
print("Batch image shape:", images.shape)   # 期望: [64, 1, 28, 28]
print("Batch label shape:", labels.shape)   # 期望: [64]

# 6. 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)