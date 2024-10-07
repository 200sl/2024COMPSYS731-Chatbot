import os.path
import queue
import random
import threading
import time
import uuid
from threading import Thread
from openai import OpenAI
from socket import *

SYSTEM_CONTEXT = "You are a psychologist at a high school counselling centre and whenever a student comes to talk to " \
                 "you, you need to decide your response based on their mood. For example, you should comfort the " \
                 "student when she feels angry or sad, encourage her when she feels lost, and motivate her to " \
                 "continue her studies when she feels happy."


# create a singleton class for result buffer
def singleton(cls):
    instance = {}

    # create a new instance if not exist, else return a existed one
    def get_instance(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]

    return get_instance


# recv result from interface via UDP
@singleton
class ResultBuffer:
    def __init__(self):
        self.resultBuffer = ""
        self.source = socket(AF_INET, SOCK_DGRAM)

    def recvThread(self):
        while True:
            results = self.source.recv(1024)
            self.resultBuffer = results.decode()
            # print("Recv From Model: ", self.resultBuffer)

    def getResult(self):
        return self.resultBuffer

    def bind(self, port):
        self.source.bind(('localhost', port))
        t = Thread(target=self.recvThread)
        t.start()


# model select and api key
MY_MODEL_SELECT = "gpt-4o-mini"
MY_API_KEY = "" if not os.path.exists("API_KEY") else open("API_KEY", "r").readline().strip()


# result receiver
RECEIVE_RESULT_PORT = 16666
resultReceiver = ResultBuffer()
resultReceiver.bind(RECEIVE_RESULT_PORT)


# get user emotion from result buffer, called by GPT4o API
def getUserEmotion():

    result = ResultBuffer().getResult()
    # print("\nFC:Get User Emotion:\n")

    return ['neutral'] if result == "" else [result]


# Class for handle chat session
class ChatSession:
    def __init__(self, apiKey=MY_API_KEY, modelSelect=MY_MODEL_SELECT):
        self.model = modelSelect
        self.client = OpenAI(api_key=apiKey)
        self.sessionId = uuid.uuid1()
        self.chatHistory = []

        # initial chat history, setup assistant role
        self.chatHistory.append(
            {
                "role": "system",  # function/assistant/user/system
                "content": SYSTEM_CONTEXT
            }
        )

        # initial function list called by GPT4o
        self.functions = [
            {
                "name": "getUserEmotion",
                "description": "Get student's emotion when they talk to you.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def clearChatHistory(self):
        self.sessionId = uuid.uuid1()
        self.chatHistory = []

        self.chatHistory.append(
            {
                "role": "system",  # function/assistant/user/system
                "content": SYSTEM_CONTEXT
            }
        )

    def switchModel(self, model):
        if model == "":
            return False

        allowModels = ['gpt-4o-mini', 'gpt-4o']

        if model not in allowModels:
            return False

        self.model = model
        return True

    def chat(self, chat, dataQueue):
        # add user's chat to chat history
        self.chatHistory.append(
            {
                "role": "user",
                "content": chat
            }
        )

        # get the stream to receive the result
        result = self.client.chat.completions.create(model=self.model,
                                                     messages=self.chatHistory, stream=True,
                                                     functions=self.functions, function_call="auto")

        # result = self.client.chat.completions.create(model=self.model,
        #                                              messages=self.chatHistory, stream=False,
        #                                              functions=self.functions, function_call="auto")

        # allData = result.choices[0].message.content

        allData = ""

        for chunk in result:

            # if the chunk is a function call, call the function and continue the chat
            if chunk.choices[0].finish_reason == "function_call":
                userEmo = getUserEmotion()
                self.chatHistory.append({
                    "role": "function",
                    "content": 'Student Emotion:' + str(userEmo),
                    "name": "getUserEmotion"
                })

                continueGenResult = self.client.chat.completions.create(model=self.model,
                                                                        messages=self.chatHistory, stream=True)

                # get result from stream
                for continueChunk in continueGenResult:
                    if type(continueChunk.choices[0].delta.content) is not str:
                        continue
                    allData += continueChunk.choices[0].delta.content
                    dataQueue.put(continueChunk.choices[0].delta.content)

                break

            if type(chunk.choices[0].delta.content) is not str:
                continue
            allData += chunk.choices[0].delta.content
            dataQueue.put(chunk.choices[0].delta.content)

        # send data to main thread
        dataQueue.put(allData)

        # save the chat history for next chat
        self.chatHistory.append(
            {
                "role": "assistant",
                "content": allData
            }
        )

    # create a new thread for non-block chat
    def chatNewThread(self, chat):
        dataQueue = queue.Queue()
        thread = Thread(target=self.chat, args=(chat, dataQueue))
        thread.start()
        return thread, dataQueue


# for test
if __name__ == "__main__":
    while True:
        print(getUserEmotion())
        time.sleep(0.3)
