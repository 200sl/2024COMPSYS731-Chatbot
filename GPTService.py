import queue
import random
import threading
import time
import uuid
from threading import Thread
from openai import OpenAI
from socket import *


def singleton(cls):
    instance = {}

    def get_instance(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]

    return get_instance


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


MY_MODEL_SELECT = "gpt-4o-mini"
MY_API_KEY = ""


RECEIVE_RESULT_PORT = 16666
resultReceiver = ResultBuffer()
resultReceiver.bind(RECEIVE_RESULT_PORT)


def getUserEmotion():
    result = ResultBuffer().getResult()

    return ['Normal'] if result == "" else [result]


class ChatSession:
    def __init__(self, apiKey=MY_API_KEY, modelSelect=MY_MODEL_SELECT):
        self.model = modelSelect
        self.client = OpenAI(api_key=apiKey)
        self.sessionId = uuid.uuid1()
        self.chatHistory = []

        self.chatHistory.append(
            {
                "role": "system",  # function/assistant/user/system
                "content": "You are a counsellor who provides counselling services to users by identifying their "
                           "feelings and emotions."
            }
        )

        self.functions = [
            {
                "name": "getUserEmotion",
                "description": "Get user's emotion via emotion recognition model",
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
                "content": "You are a counsellor who provides counselling services to users by identifying their "
                           "feelings and emotions."
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
        self.chatHistory.append(
            {
                "role": "user",
                "content": chat
            }
        )

        result = self.client.chat.completions.create(model=self.model,
                                                     messages=self.chatHistory, stream=True,
                                                     functions=self.functions, function_call="auto")

        # result = self.client.chat.completions.create(model=self.model,
        #                                              messages=self.chatHistory, stream=False,
        #                                              functions=self.functions, function_call="auto")

        # allData = result.choices[0].message.content

        allData = ""

        for chunk in result:

            if chunk.choices[0].finish_reason == "function_call":
                userEmo = getUserEmotion()
                self.chatHistory.append({
                    "role": "function",
                    "content": str(userEmo),
                    "name": "getUserEmotion"
                })

                continueGenResult = self.client.chat.completions.create(model=self.model,
                                                                        messages=self.chatHistory, stream=True)

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

        dataQueue.put(allData)

        self.chatHistory.append(
            {
                "role": "assistant",
                "content": allData
            }
        )

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
