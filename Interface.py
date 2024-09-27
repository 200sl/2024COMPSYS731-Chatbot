import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# 创建主窗口
root = tk.Tk()
root.title("Emotion Recognition Interface")
root.geometry("800x600")

# 摄像头初始化
cap = cv2.VideoCapture(0)

# 视频显示区域
video_label = tk.Label(root)
video_label.pack(padx=10, pady=10)

# 情绪识别输出
emotion_label = tk.Label(root, text="识别到的情绪的数据", fg="white", bg="blue", font=("Arial", 20))
emotion_label.pack(padx=10, pady=10)

# 情绪标签列表
emotion_labels = ['Anger', 'Happy', 'Surprise', 'Sad', 'Contempt', 'Fear', 'Disgust', 'Neutral']

# 情绪识别模型初始化
class Inferencer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 根据 AlexNet 的输入尺寸进行调整
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化参数根据预训练的模型调整
        ])

    def load_model(self, model_path):
        model = torch.load(model_path, map_location=torch.device('cpu'))
    
         # 将模型移动到正确的设备上
        model = model.to(self.device)
    
         # 设置模型为评估模式
        model.eval()
        return model

    def preprocess_image(self, frame):
        # 将摄像头帧转换为RGB格式并预处理
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)

    def predict(self, frame):
        # 预处理图像并进行预测
        image = self.preprocess_image(frame)
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        return predicted_class, confidence

# 更新模型路径
model_path = "/Users/hehahahaha/Desktop/NZ Life/UOA/S2/COMPSYS731/emotion_recognition_CNN/alexnet_face_recognition.pt"
inferencer = Inferencer(model_path)

def update_frame():
    # 从摄像头抓取一帧
    ret, frame = cap.read()
    if ret:
        # 调整帧大小以提高显示效率
        frame_resized = cv2.resize(frame, (400, 300))

        # 进行情绪识别
        predicted_class, confidence = inferencer.predict(frame_resized)

        # 检查置信度是否大于等于 0.7
        if confidence >= 0.7:
            # 获取对应的情绪标签
            predicted_emotion = emotion_labels[predicted_class]
            # 显示情绪识别结果
            emotion_label.config(text=f"Predicted Emotion: {predicted_emotion}, Confidence: {confidence:.2f}")
        else:
            # 置信度低于 0.7，显示“未检测到情绪”
            emotion_label.config(text="No emotion detected")

        # OpenCV 图像转换为 PIL 格式并显示在视频框中f
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪和缩放
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # 色彩抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 每 20 毫秒刷新一次
    video_label.after(20, update_frame)

# 开始更新视频帧
update_frame()

# 运行主循环
root.mainloop()

# 释放摄像头
cap.release()
cv2.destroyAllWindows()
