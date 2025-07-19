# pdf_chat_app.py
import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- Streamlit Page Config ---
st.set_page_config(page_title="📚 Gemini PDF Chatbot", layout="wide")
st.title("📄 Gemini-Powered PDF Chatbot")
st.markdown("Upload a PDF and ask unlimited questions about it. When done, upload another!")

# --- API Key ---
api_key = "" # Replace with your actual API key
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# --- Initialize Session State ---
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Upload PDF ---
uploaded_file = st.file_uploader("📁 Upload your PDF", type=["pdf"], key="file_upload")

# --- Process PDF on New Upload ---
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(raw_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain
# --- Detect Removed PDF ---
if uploaded_file is None and st.session_state.pdf_name is not None:
    st.session_state.pdf_name = None
    st.session_state.qa_chain = None
    st.session_state.chat_history = []

# --- Detect and Process New PDF ---
if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
    st.session_state.pdf_name = uploaded_file.name
    st.session_state.qa_chain = process_pdf(uploaded_file)
    st.session_state.chat_history = []  # reset chat history

# --- Question Interface ---
if st.session_state.qa_chain:
    question = st.text_input("💬 Ask a question about this PDF:")
    if question:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]
            st.session_state.chat_history.append((question, answer, sources))

# --- Chat History Display ---
if st.session_state.chat_history:
    st.markdown("### 🗂 Q&A History")
    for i, (q, a, s_docs) in enumerate(reversed(st.session_state.chat_history), 1):
        with st.container():
            st.markdown(f"#### 🔹 Q{i}: {q}")
            st.markdown(f"**Answer:**\n\n{a}")
            if s_docs:
                st.markdown("📄 **Source Snippets:**")
                for j, doc in enumerate(s_docs):
                    st.markdown(f"**Chunk {j+1}:** {doc.page_content[:400].strip()}...")
            st.divider()

# --- Footer ---
st.markdown("---")
st.markdown("🚀 Built with [Gemini](https://ai.google.dev/) + [LangChain](https://python.langchain.com/) + [Streamlit](https://streamlit.io)")
