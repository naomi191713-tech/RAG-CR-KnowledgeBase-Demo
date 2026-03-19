from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:7b-instruct-q5_K_M",
    temperature=0.7,
    base_url="http://localhost:11434"
)

response = llm.invoke("用一句话告诉我什么是 RAG？")
print(response.content)
