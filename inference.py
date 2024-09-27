import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import os
import cv2

class Inferencer:
    def __init__(self, model_path):

        # Determine the device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化设备（GPU 或 CPU）,在创建对象时直接准备好所有需要的资源（如模型和设备），并在之后的操作中轻松使用这些资源

        # Load the pre-trained model
        self.model = self.load_model(model_path)
        # 加载预训练模型

        # Define the image transformation pipeline. These MUST be the same as the training transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize the image with mean 0.5 and std 0.5
        ])
        # 定义图像变换管道，包括调整大小、转换为张量和归一化。

    def load_model(self, model_path):
        # Load the entire model from the specified path
        # map_location ensures the model is loaded to the correct device (CPU or GPU)
        model = torch.load(model_path, map_location=self.device)
        #加载整个模型并确保模型加载到正确的设备上。

        # Set the model to evaluation mode (important for inference)
        model.eval()
        return model
        # 将模型设置为评估模式并返回模型。
    
    def preprocess_image(self, image):
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        image = Image.fromarray(image)
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)
        # 将图像从 BGR 转换为 RGB, 将图像转换为 PIL 图像对象，应用之前定义的图像变换管道
        # 在第 0 维增加一个维度（批量维度），并将图像移动到指定设备


    def predict(self, image_path):
        # Preprocess the input image
        image = self.preprocess_image(image_path)
        # 预处理输入图像
        
        # Disable gradient calculation for inference (saves memory and computations)
        with torch.no_grad():
            # Forward pass through the model
            output = self.model(image)
            # 禁用梯度计算，进行前向传播。
            
            # Convert the output logits to probabilities using softmax
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # 使用 softmax 将输出转换为概率分布
            
            # Get the index of the class with the highest probability
            predicted_class = torch.argmax(probabilities, dim=1).item()
            # 获取概率最高的类别的索引
            
            # Get the confidence (probability) of the predicted class
            confidence = probabilities[0][predicted_class].item()
            # 获取预测类别的置信度。
        
        return predicted_class, confidence
    # 返回获取预测类别的置信度。

# Usage example:
if __name__ == "__main__":
    # Specify the path to your saved model
    model_path = "/Users/hehahahaha/Desktop/NZ Life/UOA/S2/COMPSYS731/emotion_recognition_CNN/checkpoints/8.pt"  # Adjust this to your saved model path
    # 如果脚本作为主程序运行，指定模型路径
    
    # Create an instance of the Inferencer class
    inferencer = Inferencer(model_path)
    # 创建 Inferencer 类的实例

    # Specify the path to the image you want to classify
    image_path = "/Users/hehahahaha/Desktop/NZ Life/UOA/S2/COMPSYS731/emotion_recognition_CNN/test_images/surprise/ffhq_24.png"
    # 指定要分类的图像路径
    
    # Perform the prediction
    image = cv2.imread(image_path)
    predicted_class, confidence = inferencer.predict(image)
    # 读取图像并进行预测
    
    # Print the results
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
    # 打印预测的类别和置信度