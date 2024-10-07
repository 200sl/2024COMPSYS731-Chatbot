import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

MODEL_WEIGHTS_PATH = "./source/alexnet_face_recognition.pt"

model = models.alexnet(weights=None)

fc_in_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(fc_in_features, 7)

model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root='./dataset/validate_images', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

all_labels = []
all_predictions = []
total = 0
correct = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        # predicted -= 1
        # predicted[predicted == -1] = 0
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

acc = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f2 = fbeta_score(all_labels, all_predictions, beta=2, average='weighted')

print(f"Accuracy: {acc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F2 Score: {f2:.2f}")

