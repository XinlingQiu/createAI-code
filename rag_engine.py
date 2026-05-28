import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 🚀 核心修改 1：更新了弃用的导入路径，使用最新的官方独立扩展包
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 定义提示词模板 (增加上下文关联要求)
system_prompt = (
    "你是一个博学且严谨的 AI 学习助手。你的主要任务是解答学生的疑问。\n"
    "请结合之前的对话上下文，并优先严格参考以下【参考资料】中的内容来回答。\n"
    "如果【参考资料】中能够找到完整或部分的答案，请基于资料回答，并在回答的最后单独换行加上提示语：“（💡 本回答基于您的专属课件提取）”。\n"
    "如果【参考资料】中完全没有相关内容，请运用你的通用大模型知识进行详尽解答，并在回答的最后单独换行加上提示语：“（🌐 本回答基于 AI 通用知识拓展）”。\n\n"
    "【参考资料】：\n{context}"
)

# 💡 核心升级：增加 MessagesPlaceholder 用来接收前端传来的历史对话
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def get_rag_chain(persist_directory="./chroma_db"):
    # 🚀 核心修改 2：去掉了类名中的 Bge，统一使用标准 HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    # Chroma 的初始化参数保持不变，但由于头部 import 已经更新，此处将调用最新版本
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 3. 初始化大模型 (🚨 提醒：为了工程安全，论文答辩结束后建议将明文 API_KEY 移入 .env 环境变量文件)
    API_KEY = "sk-2ee1e6f39a4c4efda8877758bb71345f"

    llm = ChatOpenAI(
        model="qwen-max",
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.3
    )

    # 4. 构建 LCEL 链
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_context(input_dict):
        """根据当前问题去数据库检索相关文档"""
        docs = retriever.invoke(input_dict["question"])

        # 💡 [探针植入]：后台监控日志，方便答辩和调试时查看底层检索了哪些内容
        print(f"\n[后台监控] 🔍 用户提问: {input_dict['question']}")
        print(f"[后台监控] 🎯 从本地 ChromaDB 召回了 {len(docs)} 个知识切片:")
        for i, doc in enumerate(docs):
            # 仅打印切片的前 60 个字符进行快速预览，避免控制台刷屏
            preview_text = doc.page_content.replace('\n', ' ')[:60]
            print(f"   片段 {i + 1}: {preview_text}...")

        return format_docs(docs)

    # 💡 核心升级：RunnablePassthrough.assign 允许我们传入字典 {question, chat_history}
    rag_chain = (
            RunnablePassthrough.assign(context=get_context)
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    print("🚀 正在测试带有记忆的 RAG 引擎...")
    chain = get_rag_chain()
    # 本地测试时不传历史记录
    for chunk in chain.stream({"question": "二叉树的定义？", "chat_history": []}):
        print(chunk, end="", flush=True)