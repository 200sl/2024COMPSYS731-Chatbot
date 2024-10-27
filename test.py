import torch
from torchvision import models
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Define the model class and load the pre-trained model
class AlexNetTester:
    def __init__(self, model_path, data_path):
        # Define image preprocessing operations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load data set
        self.test_dataset = datasets.ImageFolder(root=data_path, transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=10, shuffle=False)
        
        # Load the AlexNet model
        self.model = models.alexnet(weights=None)
        num_classes = 7  # Adjust to the actual number of categories
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        
        # Load the trained model weights
        self.model.load_state_dict(torch.load(model_path))
        
        # Move the model to the GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Output a category report for each category
        report = classification_report(all_labels, all_predictions, target_names=self.test_dataset.classes)
        print(report)

# Create a model tester instance, load model weights, and test
model_path = 'alexnet_face_recognition.pt'  # Model weight file path
data_path = '../verify_images'  # Test set data path
tester = AlexNetTester(model_path, data_path)
tester.evaluate()
