import tkinter as tk
from socket import socket, AF_INET, SOCK_DGRAM

from PIL import Image, ImageTk
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from inference import Inferencer  # Import the Inferencer class

# Create the main window using Tkinter
root = tk.Tk()
root.title("Emotion Recognition Interface")
root.geometry("800x600")

# Camera Initialization
cap = cv2.VideoCapture(0)

# Set up GUI Components
video_label = tk.Label(root)
video_label.pack(padx=10, pady=10)

# Output emotion detection data
emotion_label = tk.Label(root, text="Identified Emotion Data", fg="white", bg="blue", font=("Arial", 20))
emotion_label.pack(padx=10, pady=10)

# Define emotion labels
emotion_labels = ['Anger', 'Contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


class DataHooker:
    def __init__(self, port):
        self.sender = socket(AF_INET, SOCK_DGRAM)
        self.port = port

    def update(self, toSendData):
        serInfo = ('localhost', self.port)
        self.sender.sendto(toSendData.encode(), serInfo)


sender = DataHooker(16666)

# YOLOv8 for face detection
face_model = YOLO('./source/yolov8n-face.pt')

# Load your emotion classification model
emotion_model_path = "./source/alexnet_face_recognition.pt"
emotion_inferencer = Inferencer(emotion_model_path)


def update_frame():
    # Capture a frame from the camera
    ret, frame = cap.read()
    if ret:
        # Pass the frame to the YOLO model to detect faces
        results = face_model(frame)

        # Iterate through each detection result and perform emotion recognition
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])

                # Extract the bounding box coordinates of the detected face
                face_img = frame[y1:y2, x1:x2]

                # Initialize the emotion label and confidence
                predicted_emotion = "Unknown"
                confidence = 0.0

                if face_img.size > 0:
                    # Preprocess the face image
                    predicted_class, confidence = emotion_inferencer.predict(face_img)

                    # Check if confidence is above the threshold (0.7)
                    if confidence >= 0.7:
                        predicted_emotion = emotion_labels[predicted_class]
                        emotion_label.config(
                            text=f"Predicted Emotion: {predicted_emotion}, Confidence: {confidence:.2f}")

                        sender.update(predicted_emotion)
                    else:
                        emotion_label.config(text="No emotion detected")

                    # Draw a rectangle around the face in the frame and display the emotion label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{predicted_emotion}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        # Convert the processed OpenCV frame to PIL format and update the video display in the Tkinter window.
        frame_resized = cv2.resize(frame, (400, 300))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Refresh the frame after a fixed interval 
    video_label.after(20, update_frame)

# Repeat loop
update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
