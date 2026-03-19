from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

# ====================== 1. LLM ======================
llm = ChatOllama(
    model="qwen2.5:7b-instruct-q5_K_M",
    temperature=0.7
)

# ====================== 2. Embedding (CPU优化) ======================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ====================== 3. 知识库（自动加载文件夹版 - 绝对路径修复） ======================
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 用绝对路径，避免当前工作目录问题
knowledge_base_path = "/Users/cap/RAG_Project/knowledge_base/"

print(f"检查路径是否存在: {os.path.exists(knowledge_base_path)}")
if os.path.exists(knowledge_base_path):
    print("文件夹内文件: ", os.listdir(knowledge_base_path))
else:
    raise FileNotFoundError(f"知识库路径不存在: {knowledge_base_path}")

loader = DirectoryLoader(
    knowledge_base_path,
    glob="**/*.pdf",                   # 只匹配 .pdf（你的文件都是 pdf）
    loader_cls=PyPDFLoader,
    show_progress=True
)

docs = loader.load()

print(f"\n加载到 {len(docs)} 个原始文档")

if len(docs) > 0:
    print("第一个文档来源:", docs[0].metadata.get('source'))
    print("第一个文档内容前 200 字符:\n", docs[0].page_content[:200])
else:
    print("!!! 没有加载到任何文档 !!! 请检查：")
    print("1. 文件是否真的在 /Users/cap/RAG_Project/knowledge_base/")
    print("2. 文件名后缀是否小写 .pdf（大小写敏感）")
    print("3. PDF 是否包含可提取文本（非纯图片扫描件）")

# 继续切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
split_docs = text_splitter.split_documents(docs)

print(f"切分后得到 {len(split_docs)} 个文本块")

if len(split_docs) == 0:
    raise ValueError("切分后没有文本块！可能是 PDF 内容为空或加载失败")

# 构建向量库
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ====================== 4. RAG链 + 来源修复 ======================
template = """基于以下上下文，用专业、结构化的方式回答问题。优先使用上下文中的信息。如果涉及公式，请尽量用 LaTeX 格式写出。
如果上下文不足，诚实说明。

上下文：
{context}

问题：{question}

回答格式建议：
1. 先一句话总结核心思想
2. 详细解释机制（包括公式如果有）
3. 说明设计动机或优势
4. 标注来源

回答："""
prompt = ChatPromptTemplate.from_template(template)

def rag_answer(question):
    # ✅ 修复后的获取文档方式（新版LangChain标准）
    relevant_docs = retriever.invoke(question)
    sources = [doc.metadata.get("source", "未知") for doc in relevant_docs]
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    result = chain.invoke(question)
    
    return result.content + "\n\n📌 来源：" + " | ".join(sources)

# ====================== 5. Gradio界面（兼容 Gradio 5.x） ======================
def rag_chat(message, history):
    """
    history 是 Gradio 自动维护的 list[dict]（messages格式）
    只需返回 assistant 的回答字符串，Gradio 会自动处理
    """
    answer = rag_answer(message)
    return answer  # ← 只返回 str 即可！

demo = gr.ChatInterface(
    fn=rag_chat,
    title="🎉 我的RAG复试项目Demo - Gradio 5.x 兼容版",
    description="基于LangChain + Ollama + Chroma 的 RAG 知识库问答系统（Mac Intel 版）\n快问问复试相关问题吧！",
    chatbot=gr.Chatbot(height=500),  # 可以自定义高度等
    textbox=gr.Textbox(placeholder="例如：这个项目有什么亮点？  或  复试会考什么？  或  RAG是什么？"),
    # 删除 clear_btn、submit_btn 等参数！Gradio 5.x 内置了清空按钮（🗑️图标）
    # 如果想自定义清空按钮文本，需要用 Blocks 手动实现（见下面备选）
)

print("✅ Demo已更新！正在启动...")
demo.launch(share=False, debug=True)  # 加 debug=True 方便看详细日志