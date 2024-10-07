from GPTService import ChatSession
import json
import speech_recognition as sr

if __name__ == '__main__':
    chatSession = ChatSession()
    recognizer = sr.Recognizer()

    print("==================================")
    print("Welcome to EduBot Lab")
    print("Model Version: gpt-4o-mini")
    print("Type '.end' to exit, '.help' for help")
    print("==================================")

    while True:
        msg: str = input("You: ")

        # Check User Input and execute the corresponding command
        if msg is None:
            continue

        if msg == ".info":
            print("Bot:")
            print(f"\t Model Version: {chatSession.model}")
            continue

        # Switch the GPT API Model
        if msg.startswith(".swmodel"):
            msgs = msg.strip().split(" ")

            if chatSession.switchModel(msgs[-1]):
                print(f"Bot: Model switched to {msgs[-1]}")
            else:
                print("Bot: Invalid model version")
            continue

        if msg == ".end":
            print("Bot: BYE")
            break

        if msg == ".help":
            print("Bot:")
            print("\t .end: Exit")
            print("\t .reset: Reset chat session")
            print("\t .multi: Enable multi-line mode")
            print("\t .help: Show this help")
            print("\t .swmodel <model>: Switch model version\n\t\tAvailable models: gpt-4o-mini, gpt-4o")
            continue

        if msg == ".reset":
            chatSession.clearChatHistory()
            print("Bot: Chat session reset")
            continue

        # the command cannot handle multi-line input, so add a multi-line mode
        if msg == ".multi":
            multi_buffer = ""
            print("Bot: Multi-line mode enabled, please input until we meet .multiend")

            while True:
                dat = input()
                if dat == ".multiend":
                    msg = multi_buffer
                    break
                else:
                    multi_buffer += dat + "\n"

        if msg == ".s":
            # Speech recognition, using vosk model
            with sr.Microphone() as source:
                print("Bot: Speak now")
                audio = recognizer.listen(source, timeout=5)
                try:
                    # get the text from the audio, and pass to the chat session
                    text = json.loads(recognizer.recognize_vosk(audio))
                    print("You: ", text['text'])
                    msg = text['text']
                except sr.UnknownValueError:
                    print("Bot: Sorry, I did not get that")
                    continue
                except sr.RequestError as e:
                    print("Bot: Sorry, I am not able to process your request at the moment; {0}".format(e))
                    continue

        if msg == "":
            print("Bot: Please type something or .help for more info")
            continue

        t, dataQ = chatSession.chatNewThread(msg)
        print("Bot: ", end="", flush=True)
        while t.is_alive():
            if not dataQ.empty():
                print(dataQ.get(), end="", flush=True)
        print()
        print()

    input("Press Enter to exit")

