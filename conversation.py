from agent_from_scratch.chat_models import ChatModel


model = ChatModel.from_settings()


conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to Vietnamese."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "Tôi yêu lập trình."},
    {"role": "user", "content": "Translate: I love building applications."},
]

response = model.invoke(messages=conversation)
print(response)
