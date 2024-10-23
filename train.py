import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from earlystopping import EarlyStopping
import timm

RESULT_FILE_NAME = 'resnet18_rafdb_highLr_2.txt'

class Trainer:
    def __init__(self):
        # Define image preprocessing operations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load data set
        self.train_dataset = datasets.ImageFolder(root='dataset/train_images', transform=transform)
        self.test_dataset = datasets.ImageFolder(root='dataset/test_images', transform=test_transform)

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        # Load the pre-trained AlexNet model
        self.model = timm.create_model('timm/resnet18.a1_in1k', pretrained=False)

        # Modify the last layer for the number of categories
        num_classes = 7  # Adjusted for the number of categories in the dataset

        # For Resnet18 transfer learning
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        # num_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_features, num_classes)

        # Define loss functions and optimizers
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Define learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=16, gamma=0.1)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, num_epochs, early_stopping):
        best_accuracy = 0.0  # Initialize best accuracy

        train_acc_list = []
        test_acc_list = []

        train_loss_list = []
        test_loss_list = []

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            correct = 0
            total = 0
            step = 0

            start_time = time.time()
            # Training loop
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                step += 1

                train_loss = self.criterion(outputs, labels)
                train_loss.backward()
                self.optimizer.step()
                running_loss += train_loss.item()

                if step % 10 == 0:
                    print(f'Step:{step} / {len(self.train_loader)}')

            print(f'Epoch [{epoch + 1}/{num_epochs}], time cost: {time.time() - start_time:.2f}s')

            train_loss_list.append(running_loss / len(self.train_loader))
            train_acc_list.append(correct / total)

            # Print the current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            print(f'Learning rate: {current_lr}')

            # Adjust learning rate
            self.scheduler.step()

            # Evaluate the model on the test set
            accuracy, loss_test = self.evaluate(self.criterion)

            test_acc_list.append(accuracy)
            test_loss_list.append(loss_test)

            # Save the model if the accuracy is the best we've seen so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), 'alexnet_face_recognition.pt')
                print(f"Saved Best Model with Accuracy: {best_accuracy}%")

            # Call the early stopping
            early_stopping(accuracy)

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return train_acc_list, test_acc_list, train_loss_list, test_loss_list

    def evaluate(self, loss_fn):
        self.model.eval()
        correct = 0
        total = 0

        total_loss = 0.0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                validate_loss = loss_fn(outputs, labels)
                total_loss += validate_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        rt_loss = total_loss / len(self.test_loader)
        print(f'Validation Accuracy: {accuracy}%')
        return accuracy, rt_loss


# Create a Trainer instance
trainer = Trainer()

# Create an EarlyStopping instance with patience=5
early_stopping = EarlyStopping(patience=10, delta=0.05)

# Start training with early stopping
tra_acc, test_acc, tra_loss, test_loss = trainer.train(64, early_stopping)

result_file = open(RESULT_FILE_NAME, 'w')

result_file.write('train_acc:')
for acc in tra_acc:
    result_file.write(f'{acc}, ')
result_file.write('\n')

result_file.write('test_acc:')
for acc in test_acc:
    result_file.write(f'{acc}, ')
result_file.write('\n')

result_file.write('train_loss:')
for loss in tra_loss:
    result_file.write(f'{loss}, ')
result_file.write('\n')

result_file.write('test_loss:')
for loss in test_loss:
    result_file.write(f'{loss}, ')
result_file.write('\n')

result_file.close()
