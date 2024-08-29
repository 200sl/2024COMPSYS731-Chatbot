import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import os

# 创建主窗口
root = tk.Tk()
root.title("Emotion Recognition Interface")
root.geometry("1200x800")

# 模拟聊天会话
class ChatSession:
    def __init__(self):
        self.model = "gpt-4o-mini"

    def chatNewThread(self, msg):
        # 模拟聊天机器人回复
        response = self.simulate_chatbot_response(msg)
        return response

    def simulate_chatbot_response(self, msg):
        # 这里可以根据输入消息返回一个模拟的回复
        if "hello" in msg.lower():
            return "Hello! How can I assist you today?"
        elif "bye" in msg.lower():
            return "Goodbye! Have a great day!"
        else:
            return "I'm sorry, I didn't understand that. Can you please rephrase?"

# 初始化聊天会话
chatSession = ChatSession()

# 左侧区域：ChatBot 界面
frame_left = tk.Frame(root, bg='blue', width=400, height=600)
frame_left.grid(row=0, column=0, padx=10, pady=10)
frame_left.grid_propagate(False)

chat_label = tk.Label(frame_left, text="ChatBot", fg="white", bg="blue", font=("Arial", 20))
chat_label.pack(pady=10)

# 聊天记录显示框
chat_display = tk.Text(frame_left, width=40, height=20)
chat_display.pack(pady=10)

# 用户输入框
user_input = tk.Entry(frame_left, width=40)
user_input.pack(pady=10)

# 发送按钮
def send_message():
    msg = user_input.get()
    if msg:
        chat_display.insert(tk.END, f"You: {msg}\n")
        user_input.delete(0, tk.END)
        response = chatSession.chatNewThread(msg)
        chat_display.insert(tk.END, f"Bot: {response}\n")

send_button = tk.Button(frame_left, text="Send", command=send_message)
send_button.pack(pady=10)

# 下方区域：实时视频显示
frame_bottom = tk.Frame(root, bg='blue', width=400, height=200)
frame_bottom.grid(row=1, column=0, padx=10, pady=10)
frame_bottom.grid_propagate(False)

video_label = tk.Label(frame_bottom)
video_label.pack()

# 右侧区域：情绪识别输出
frame_right = tk.Frame(root, bg='blue', width=700, height=800)
frame_right.grid(row=0, column=1, rowspan=2, padx=10, pady=10)
frame_right.grid_propagate(False)

emotion_label = tk.Label(frame_right, text="识别到的情绪的数据", fg="white", bg="blue", font=("Arial", 20))
emotion_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# 摄像头初始化
cap = cv2.VideoCapture(0)

# 情绪识别模型初始化
class Inferencer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)

    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        return predicted_class, confidence

model_path = "COMPSYS731/emotion_recognition_CNN/checkpoints/8.pt"
inferencer = Inferencer(model_path)

def update_frame():
    # 从摄像头抓取一帧
    ret, frame = cap.read()
    if ret:
        # OpenCV 图像转换为 PIL 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # 显示在视频框中
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # 进行情绪识别
        predicted_class, confidence = inferencer.predict(frame)
        emotion_label.config(text=f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

    # 每 10 毫秒刷新一次
    video_label.after(10, update_frame)

# 开始更新视频帧
update_frame()

# 运行主循环
root.mainloop()

# 释放摄像头
cap.release()
cv2.destroyAllWindows()