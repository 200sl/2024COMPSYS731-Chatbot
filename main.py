from GPTService import ChatSession


if __name__ == '__main__':
    chatSession = ChatSession()

    print("==================================")
    print("Welcome to EduBot Lab")
    print("Model Version: gpt-4o-mini")
    print("Type '.end' to exit, '.help' for help")
    print("==================================")

    while True:
        msg: str = input("You: ")

        if msg is None:
            continue

        if msg == ".info":
            print("Bot:")
            print(f"\t Model Version: {chatSession.model}")
            continue

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

