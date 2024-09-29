import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
from torch.utils.data import DataLoader
from torchvision import models
from dataset import CustomImageDataset
import os

    # Load the pre-trained AlexNet model
model = models.alexnet(weights=True)
model.load_state_dict(torch.load('alexnet-owt-7be5be79.pth', weights_only=True))
# model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)


for param in model.features.parameters():  
     param.requires_grad = False

     # Modify the last layer to accommodate the number of categories for expression recognition
num_classes = 8  # Adjusted for the number of data set categories
model.classifier[6] = nn.Linear(4096, num_classes)

    # Define loss functions and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # Define the learning rate scheduler, step_size=30 indicates that the learning rate decays once every 30 epochs,
    # and gamma=0.1 indicates that the decays are 0.1 times as large as before
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Define image preprocessing operations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪和缩放
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # 色彩抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

   # Load data set
train_dataset = CustomImageDataset(root_dir='C:/Glory/UoA-Master/Cos731/AlexNet/train_images', transform=transform)
test_dataset = CustomImageDataset(root_dir='C:/Glory/UoA-Master/Cos731/AlexNet/test_images', transform=transform)

# train_dataset = datasets.ImageFolder(root='C:/Glory/UoA-Master/Cos731/AlexNet/train_images', transform=transform)
# test_dataset = datasets.ImageFolder(root='C:/Glory/UoA-Master/Cos731/AlexNet/test_images', transform=transform)

   # Create a data loader
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

   # Move the model to the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


num_epochs = 30
model.train()

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

            # Print the current learning rate
            current_lr = scheduler.get_last_lr()[0]
            print(f'Learning rate: {current_lr}')


            # Adjusted learning rate
            scheduler.step()

            # Evaluation model
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
