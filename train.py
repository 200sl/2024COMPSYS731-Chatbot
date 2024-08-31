import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
import os

# class Trainer:
#    def _init_(a, save_path):

    # 加载预训练的 AlexNet 模型
model = models.alexnet(weights=True)
model.load_state_dict(torch.load('alexnet-owt-7be5be79.pth', weights_only=True))
    # models.load_state_dict(torch.load('alexnet-owt-7be5be79.pth'), weights=True)


     # 修改最后一层，适应表情识别的类别数量
num_classes = 8  # 根据数据集类别数调整
model.classifier[6] = nn.Linear(4096, num_classes)

    # 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 定义图像预处理操作
transform = transforms.Compose([
         transforms.Resize((224, 224)),  # AlexNet 输入尺寸
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

   # 加载数据集
train_dataset = datasets.ImageFolder(root='C:/Glory/UoA-Master/Cos731/AlexNet/train_images', transform=transform)
test_dataset = datasets.ImageFolder(root='C:/Glory/UoA-Master/Cos731/AlexNet/test_images', transform=transform)

   # 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

   # 将模型移动到 GPU 上（如果有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 
num_epochs = 10
model.train()
#    def train(a, num_epochs):

for epoch in range(num_epochs):

            
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # 评估模型
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

print(f'Accuracy: {100 * correct / total}%')

torch.save(model.state_dict(), 'alexnet_face_recognition.pt')

model.load_state_dict(torch.load('alexnet_face_recognition.pt'))

#os.makedirs(save_path, exist_ok=True)
#torch.save(self.model, save_path + os.sep + str(epoch) + ".pt")

# 用新图像进行推理...