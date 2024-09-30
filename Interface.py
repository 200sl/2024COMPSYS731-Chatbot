import tkinter as tk
from socket import socket, AF_INET, SOCK_DGRAM
from PIL import Image, ImageTk
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from inference import Inferencer  # Import the Inferencer class

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
emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


class DataHooker:
    def __init__(self, port):
        self.sender = socket(AF_INET, SOCK_DGRAM)
        self.port = port

    def update(self, toSendData):
        serInfo = ('localhost', self.port)
        self.sender.sendto(toSendData.encode(),serInfo)


sender = DataHooker(16666)

# YOLOv8 for face detection
face_model = YOLO('/Users/hehahahaha/Desktop/NZ Life/UOA/S2/COMPSYS731/emotion_recognition_CNN/yolov8n-face.pt')

# Load your emotion classification model
emotion_model_path = "/Users/hehahahaha/Desktop/NZ Life/UOA/S2/COMPSYS731/emotion_recognition_CNN/checkpoints/8.pt"
emotion_inferencer = Inferencer(emotion_model_path)


# 情绪识别模型初始化
class Inferencer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        model = model.to(self.device)
        model.eval()  # Set the model to evaluation mode
        return model

    def preprocess_image(self, frame):
        # Convert the frame to RGB and apply transforms
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)

    def predict(self, face_img):
        # Preprocess image and run emotion prediction
        image = self.preprocess_image(face_img)
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        return predicted_class, confidence

def update_frame():
    # 从摄像头抓取一帧
    ret, frame = cap.read()
    if ret:
        # 运行YOLO进行人脸检测
        results = face_model(frame)
        
        # 遍历每个检测结果并进行情绪识别
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])

                # 提取人脸区域
                face_img = frame[y1:y2, x1:x2]
                
                # 初始化情绪标签的默认值
                predicted_emotion = "Unknown"
                confidence = 0.0
                
                if face_img.size > 0:
                    # 进行情绪分类预测
                    predicted_class, confidence = emotion_inferencer.predict(face_img)

                    # 检查置信度是否大于等于 0.7
                    if confidence >= 0.7:
                        predicted_emotion = emotion_labels[predicted_class]
                        emotion_label.config(text=f"Predicted Emotion: {predicted_emotion}, Confidence: {confidence:.2f}")

                        sender.update(predicted_emotion)
                    else:
                        emotion_label.config(text="No emotion detected")

                    # 绘制边框和情绪标签
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{predicted_emotion}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        # OpenCV 图像转换为 PIL 格式并显示在 Tkinter 界面中
        frame_resized = cv2.resize(frame, (400, 300))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # 每 20 毫秒刷新一次
    video_label.after(1000, update_frame)


# 开始更新视频帧
update_frame()

# 运行主循环
root.mainloop()

# 释放摄像头
cap.release()
cv2.destroyAllWindows()
