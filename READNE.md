# RAG 知识库智能问答系统 - 计算机考研复试项目

基于 LangChain + Ollama + Chroma + Gradio 实现的本地 RAG 系统，用于考研复试项目展示。

## 项目亮点
- 使用 Qwen2.5-7B-Instruct 本地量化模型（Mac Intel 兼容）
- 支持 PDF/TXT 自动加载与 chunk 切分
- 检索增强生成（RAG），有效缓解幻觉
- Gradio Web 界面，支持多轮对话
- 知识库包含《Attention is All You Need》、《Retrieval-Augmented Generation》论文等

## 技术栈
- LLM: Ollama + Qwen2.5-7B-Instruct (Q5_K_M)
- Embedding: BAAI/bge-small-zh-v1.5
- Vector DB: Chroma
- RAG 框架: LangChain
- 前端: Gradio

## 运行方式
1. 安装依赖：`pip install -r requirements.txt`
2. 启动 Ollama 并拉取模型：`ollama run qwen2.5:7b-instruct-q5_K_M`
3. 运行：`python rag_demo.py`

## 未来优化方向
- 添加 rerank 模块（bge-reranker-base）
- 支持历史对话记忆
- 部署到服务器 / Docker
- 多模态扩展

欢迎老师/同学 star 或 fork！