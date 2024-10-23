# ReadMe

## 1.Package Installing
- apt install openai  
- apt install SpeechRecognition  
- apt install vosk  
- apt install PyAudio  

if you are using MacOS, make sure you have installed PortAudio before you install PyAudio, otherwise do
brew install portaudio

## 2.Setup API Key
In GTPService.py, line 44, put the OpenAI API Key to enable the GTP Models, or save the api key in a file named API_KEY and put the file in the same directory with project root.

## 3.Running
You need to run two processes, main.py and interface.py.  
The file main.py is used to implement the ChatBot with emotion recognition model and Speech to Text input, the other file interface.py is used to running the recognition model, sending model result to the ChatBot and opening a gui for user to see the result of the model. 
