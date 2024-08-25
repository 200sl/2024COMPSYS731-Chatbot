import queue
import random
import uuid
from threading import Thread

from openai import OpenAI

MY_MODEL_SELECT = "gpt-4o-mini"
MY_API_KEY = ""


def getUserEmotion():
    results = ['高兴', '伤心', '愤怒', '失望']
    result = random.Random().choice(results)
    print(f"用户情绪：{result}")
    return [result]


class ChatSession:
    def __init__(self, apiKey=MY_API_KEY, modelSelect=MY_MODEL_SELECT):
        self.model = modelSelect
        self.client = OpenAI(api_key=apiKey)
        self.sessionId = uuid.uuid1()
        self.chatHistory = []

        self.chatHistory.append(
            {
                "role": "system",  # function/assistant/user/system
                "content": "你是一个心理咨询师，通过识别用户的情感和情绪，为用户提供心理咨询服务。"
            }
        )

        self.functions = [
            {
                "name": "getUserEmotion",
                "description": "获取用户情绪",
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
                "content": "你是一个心理咨询师，通过识别用户的情感和情绪，为用户提供心理咨询服务。"
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

