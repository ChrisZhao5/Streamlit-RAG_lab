import streamlit as st
import os
import tempfile

# --- 导入库 ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    # 使用 HuggingFace 本地模型 (免费、稳定、不用联网)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # 使用 Google Gemini 回答问题
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
except ImportError as e:
    st.error(f"环境缺少库，请运行: pip install sentence-transformers \n 错误详情: {e}")
    st.stop()

# 1. 页面设置
st.set_page_config(page_title="Solaria Labs RAG Demo", layout="wide")
st.title("🤖 Bohan's RAG Prototype (Powered by Gemini 3.0)")

# 2. 侧边栏
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Google API Key", type="password", help="Starts with AIza...")
    st.markdown("---")
    st.markdown("**Tech Stack:**\n- **Embeddings:** HuggingFace (Local)\n- **LLM:** Gemini 3 Flash (Cloud)\n- **Vector DB:** FAISS")

# 3. 文件上传
uploaded_file = st.file_uploader("Upload Document (PDF only)", type="pdf")

if uploaded_file and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # --- 状态 A: 处理文档 (本地运行) ---
    if "vectorstore" not in st.session_state:
        with st.spinner("🚀 Processing Document with Local CPU..."):
            try:
                # A. 保存临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # B. 加载
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()

                # C. 切分
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)

                # D. 向量化 (使用本地模型 all-MiniLM-L6-v2)
                # 这一步不需要 Key，完全在本地跑
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # 存入 Session State 防止每次提问都重跑
                st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                
                st.success("✅ Knowledge Base Ready!")
                os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"Error initializing RAG: {e}")
                st.stop()

    # --- 状态 B: 问答界面 (调用云端 Gemini 3) ---
    query = st.text_input("What would you like to know?")

    if query:
        with st.spinner("🤖 Gemini 3 is Thinking..."):
            try:
                # E. 初始化 LLM (关键修改点！)
                # 使用你列表里的真实模型名称
                llm = ChatGoogleGenerativeAI(
                    model="models/gemini-3-flash-preview", 
                    temperature=0
                )
                
                # F. 构建检索链
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(),
                    return_source_documents=True
                )
                
                # G. 提问
                result = qa_chain.invoke({"query": query})
                
                st.markdown("#### Answer:")
                st.info(result["result"])
                
                with st.expander("Show Source Context"):
                    for doc in result["source_documents"]:
                        st.text(f"Page {doc.metadata.get('page', '?')}:")
                        st.write(doc.page_content[:300] + "...")

            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("👈 Enter Google API Key to start.")