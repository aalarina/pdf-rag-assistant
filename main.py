from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

# -----------------------
# 1. LOAD ENV
# -----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------
# 2. LOAD PDF
# -----------------------
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# -----------------------
# 3. SPLIT INTO CHUNKS
# -----------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

# -----------------------
# 4. CREATE VECTOR DB
# -----------------------
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore

# -----------------------
# 5. CREATE LLM
# -----------------------
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile"
    )

# -----------------------
# 6. QUERY PIPELINE
# -----------------------
def ask_question(vectorstore, llm, query):
    docs = vectorstore.similarity_search(query, k=3)

    context = "\n\n".join(
        [f"PAGE {doc.metadata.get('page')}:\n{doc.page_content}" for doc in docs]
    )

    prompt = f"""
You are a strict document assistant.

Answer ONLY using the context below.

If the answer is not explicitly present in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    return response.content

# -----------------------
# 7. MAIN FLOW
# -----------------------
if __name__ == "__main__":
    pdf_path = "sample.pdf"

    documents = load_documents(pdf_path)
    chunks = split_documents(documents)

    vectorstore = create_vectorstore(chunks)
    llm = get_llm()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer = ask_question(vectorstore, llm, query)
        print("\nAnswer:\n", answer)
