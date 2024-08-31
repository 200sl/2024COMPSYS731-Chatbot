import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import torch.nn as nn

class Inferencer:
    def __init__(self, model_path):
        # Determine the device (GPU if available, otherwise CPU)
    #    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the pre-trained model
        self.model = self.load_model(model_path)
        
        # Define the image transformation pipeline. These MUST be the same as the training transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # AlexNet 输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        # Load the entire model from the specified path
        # map_location ensures the model is loaded to the correct device (CPU or GPU)
        model = models.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 8)
    #    state_dict = torch.load('path_to_model_weights.pth')
    #    model.load_state_dict(state_dict)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    #    model = torch.load(model_path, map_location=self.device)
        
        # Set the model to evaluation mode (important for inference)
        model.eval()
        return model
    
    def preprocess_image(self, image):
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        image = Image.fromarray(image)
        image = self.transform(image)
        return image.unsqueeze(0)
    


    def predict(self, image_path):
        # Preprocess the input image
        image = self.preprocess_image(image_path)
        
        # Disable gradient calculation for inference (saves memory and computations)
        with torch.no_grad():
            # Forward pass through the model
            output = self.model(image)
            
            # Convert the output logits to probabilities using softmax
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Get the index of the class with the highest probability
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # Get the confidence (probability) of the predicted class
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence

# Usage example:
if __name__ == "__main__":
    # Specify the path to your saved model
    model_path = "alexnet_face_recognition.pt"  # Adjust this to your saved model path
    
    # Create an instance of the Inferencer class
    inferencer = Inferencer(model_path)

    # Specify the path to the image you want to classify
    image_path = "test_images/anger/image0000333.jpg"
    
    # Perform the prediction
    image = cv2.imread(image_path)
    predicted_class, confidence = inferencer.predict(image)
    
    emotion_labels = ['Anger', 'Happy','Surprise','Sad', 'Contempt', 'Fear',  'Disgust', 'Neutral', ] 
    emotion_label = emotion_labels[predicted_class]
    # Print the results
    print(f"Predicted class: {predicted_class}, Emotion: {emotion_label}, Confidence: {confidence:.2f}")
