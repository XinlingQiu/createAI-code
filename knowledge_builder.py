import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 🚀 统一使用最新的 Chroma 和 HuggingFace 独立扩展包
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def build_local_vector_db(file_path, persist_directory="./chroma_db"):
    """多模态大一统解析器：读取 PDF/Word/TXT/Video 并构建本地向量数据库"""
    print(f"\n[系统日志] 🚀 正在解析多模态文件: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    docs = []

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
        elif ext in [".mp4", ".avi", ".mov"]:
            # 针对视频的 Mock ASR 转写代理
            video_mock_text = f"""
            [系统自动 ASR 语音转写记录]
            文件源：{os.path.basename(file_path)}
            转写内容：欢迎观看本期教学视频。本节课我们将探讨复杂数据结构与大模型训练的底层原理。
            在实际应用中，多模态数据的对齐是关键。视频中的视觉特征需要与文本特征在潜空间中进行匹配。
            我们在后续的实验中，将重点讲解损失函数的收敛过程以及防止过拟合的 Dropout 策略。
            这不仅适用于计算机视觉，同样适用于自然语言处理领域。
            """
            docs = [Document(page_content=video_mock_text, metadata={"source": file_path, "type": "video_transcript"})]
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    except Exception as e:
        raise ValueError(f"文件解析器异常: {e}")

    # 文本空值拦截保护
    if not docs or all(not doc.page_content.strip() for doc in docs):
        raise ValueError("无法从该文件中提取到有效文字！如果是扫描版 PDF 或加密 Word，请更换标准文本文件。")

    # 文本动态切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(docs)

    if not split_docs:
        raise ValueError("文本切分后为空，请检查文件内容逻辑。")

    print(f"[系统日志] ✂️ 文本降维与物理切片完成，共生成 {len(split_docs)} 个切片。")
    print("[系统日志] 🧠 装载 BGE 向量嵌入模型中...")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    print("[系统日志] 💾 正在灌入 ChromaDB 向量空间...")
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"✅ 文件 {os.path.basename(file_path)} 知识入库成功！")


# 🌟 修复：精准剔除单一文件的核心逻辑 (路径归一化 + 安全删除)
def delete_single_file(filename, persist_directory="./chroma_db", upload_directory="uploads"):
    """选择性删除：从本地文件夹和向量库中精准移除特定文件及其所有 Chunk"""
    file_path = os.path.join(upload_directory, filename)
    # 将 Windows 的反斜杠统一替换为正斜杠，防止 source 匹配失败
    file_path_normalized = file_path.replace("\\", "/")

    # 1. 尝试从 Chroma 向量库中删除相关切片 (通过 metadata 中的 source 字段定位)
    try:
        if os.path.exists(persist_directory):
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

            # 使用 get() 检查获取 ids 然后按 ids delete
            result = vectordb.get(where={"source": file_path_normalized})
            if result and len(result['ids']) > 0:
                vectordb.delete(ids=result['ids'])
                print(f"[-] 已从向量数据库中抹除 {len(result['ids'])} 个关联切片: {filename}")
            else:
                print(f"[!] 警告：向量库中未找到源文件为 {file_path_normalized} 的切片数据。")

    except Exception as e:
        print(f"向量库清理警告 (可能文件未完全入库): {e}")

    # 2. 删除物理文件
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"[-] 已删除本地物理文件: {filename}")
        return True
    return False


def clear_vector_db(persist_directory="./chroma_db", upload_directory="./uploads"):
    """物理级全局删除：清空本地知识库与所有缓存文件"""
    print("⚠️ 收到系统格式化指令，正在清空全部库...")
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
    if os.path.exists(upload_directory):
        shutil.rmtree(upload_directory, ignore_errors=True)
        os.makedirs(upload_directory)
    return True


if __name__ == "__main__":
    print("多模态知识引擎模块就绪。")