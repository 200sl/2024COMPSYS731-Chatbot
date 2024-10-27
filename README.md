
# ChatGPT-powered emotion recognition robot

## Project Overview
This project develops an emotion recognition robot that combines **ChatGPT** and **Emotion detection models** to recognize user emotions and generate personalized responses. Emotion recognition captures user facial expressions in real time through a camera, and combined with the dialogue generation provided by ChatGPT, the robot can dynamically adapt to the user's emotional state, which is particularly suitable for education and emotional interaction scenarios.
---

## Project structure
```
project_root/
│
├── main.py              # Main program entry of the project, start emotion recognition and chat function
├── train.py                 # Script for training emotion detection model
├── inference.py             # Model inference script, real-time emotion detection
├── earlystopping.py     # Implement early stopping mechanism to prevent model overfitting
├── GPTService.py       # Interaction module with ChatGPT API, used to generate emotion-based responses
├── test.py                 # For trained model performance testing
├── Interface.py           # User interface and interactive interface, display and receive user emotions and dialogues
├── yolov8n-face.pt                # Face image capture processing model
├── alexnet_face_recognition.pt      # Trained emotion detection model
├── alexnet-owt-7be5be79.pth       # Pre-trained alexnet model
├── train_images/               # Training set images
├── test_images/                # Test set images
├── verify_images/              # Verify_images
│
└── README.md                  # Project description document
```

---

## Installation guide

### 1. Clone the project
```bash
git clone https://github.com/200sl/2024COMPSYS731-Chatbot.git
cd 2024COMPSYS731-Chatbot
```

### 2. Install dependencies

```bash
pip install opencv-python==4.7.0.72 
pip install ultralytics 
pip install numpy==1.24.4 
pip install openai
```

Dependencies include:
- `opencv-python==4.7.0.72`：used for camera image capture and processing.
- `ultralytics`：provides YOLO model support.
- `numpy==1.24.4`：used for numerical calculations.
- `openai`：used to access OpenAI's ChatGPT API

---

## How to use

### 1. Train the emotion detection model

The emotion detection model is trained based on `train.py`. Seven emotions (such as happiness, sadness, anger, etc.) are recognized through facial expressions, and the trained model is saved to `alexnet_face_recognition.pt`.

#### （1）Data preparation
Create `train_images/` and `test_images/` folders, and store the training and test images in subfolders by category.

#### （2）Start training
Run the following command to train the model:

```bash
python train.py
```

The training script contains **Early Stopping** to optimize the training process and improve the model's adaptability to class imbalance.

#### （3）Testing and evaluation

Run `test.py` to test the performance of the trained model

Create a `verify_images` folder and store the test images in subfolders by category.

Run the following command to train the model:

```bash
python test.py
```

### 2. Real-time emotion detection and dialogue generation

The emotion detection and chat modules are integrated in `main.py`. The system captures images through the camera and classifies emotions, and then calls ChatGPT to generate corresponding dialogues based on the emotional state.

#### （1）API configuration
Add the API key to the `MY_API_KEY` line in the `GPTService.py` file:
```
MY_API_KEY=""
```

#### （2）Run the main program
Start `Interface.py` for real-time emotion detection:
Start `main.py` for real-time chatbot interaction:

```bash
python Interface.py
python main.py
```

When the program is run, a real-time emotion monitoring window and a Chatbot chat window will appear.
The emotion recognition window will display the user's current emotion.
Users can ask random questions to Chatbot or just chat, and the robot will generate adaptive conversation content based on the real-time detected emotions, such as providing comfort when "sadness" is detected, and encouragement when "happy" is detected, and give corresponding learning guidance.

Users can also switch ChatGPT versions, use `gpt-4o-mini` or `gpt-4o`, just tell Chatbot

Users can also switch voice chat or text input, just tell Chatbot

---

### 3. File Description

- **train.py**：Emotion detection model training script, integrated with Early Stopping.
- **inference.py**：Emotion detection inference module, responsible for real-time image emotion recognition.
- **GPTService.py**：Module that interacts with ChatGPT API, generating appropriate conversation content based on emotions.
- **Interface.py**：Defines the interactive interface with users, displays emotions and chat response content.
- **earlystopping.py**：Implementation of early stopping mechanism to avoid model overfitting.
- **test.py**：Emotion detection model performance Test script.
  
---

## Contact information

For questions or suggestions, please contact the project maintainer:**mmmmanticore@gmail.com**
