import torch
import torch.nn as nn
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from dataset import CustomImageDataset
from earlystopping import EarlyStopping
from sklearn.metrics import precision_score, recall_score, fbeta_score


class Trainer:
    def __init__(self):
        # Define image preprocessing operations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
                    
        # Load data set
        self.train_dataset = datasets.ImageFolder(root='../AlexNet/train_images', transform=transform)
        self.test_dataset = datasets.ImageFolder(root='../AlexNet/test_images', transform=transform)
        print(self.train_dataset.classes)

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=10, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=10, shuffle=False)

        # Load the pre-trained AlexNet model
        self.model = models.alexnet(weights=None)
        self.model.load_state_dict(torch.load('alexnet-owt-7be5be79.pth', weights_only=True))

        # Modify the last layer for the number of categories
        num_classes = 7  # Adjusted for the number of categories in the dataset
        self.model.classifier[6] = nn.Linear(4096, num_classes)

        # Define loss functions and optimizers
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)

        # Define learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, num_epochs, early_stopping):
        best_accuracy = 0.0  # Initialize best accuracy

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            # Training loop
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.train_loader):.4f}')

            # Print the current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            print(f'Learning rate: {current_lr}')

            # Adjust learning rate
            self.scheduler.step()

            # Evaluate the model on the test set
            accuracy = self.evaluate()

            # Save the model if the accuracy is the best we've seen so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), 'alexnet_face_recognition.pt')
                print(f"Saved Best Model with Accuracy: {best_accuracy}%")

            # Call the early stopping
            early_stopping(accuracy)

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Save all labels and predictions
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy}%')

        # Calculate Precision, Recall, and F2-Score
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f2 = fbeta_score(all_labels, all_predictions, beta=2, average='weighted')

        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F2-Score: {f2:.4f}')

        return accuracy


# Create a Trainer instance
trainer = Trainer()

# Create an EarlyStopping instance with patience=10
early_stopping = EarlyStopping(patience=20, delta=0)

# Start training with early stopping
trainer.train(200, early_stopping)
