from langchain_community.chat_models import ChatOllama

def get_llm():
    return ChatOllama(
        model="phi3:latest",
        base_url="http://ollama:11434",
        temperature=0
    )