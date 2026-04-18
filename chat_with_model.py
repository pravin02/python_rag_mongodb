from ollama import chat;

while True:
    query = input("\n\n You can ask me anything >> ");
    model_response = chat(
        model= "gemma4",
        messages= [{"role": "Teacher", "content": query}]
    );
    print("\n",model_response.message.content);
