import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import timm

MODEL_WEIGHTS_PATH = "./alexnet_face_recognition.pt"

model = timm.create_model('resnet18', pretrained=False)

model.fc = torch.nn.Linear(model.fc.in_features, 7)

model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cuda'))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root='./dataset/test_images', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

class_name = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
all_labels = []
all_predictions = []
total = 0
correct = 0

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(classification_report(all_labels, all_predictions, target_names=class_name))
print(f"Accuracy: {correct / total:.2f}")

