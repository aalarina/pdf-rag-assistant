from dotenv import load_dotenv
import os
import spaces

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever

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
    # Let's increase `chunk_size` to 800 so that the chunks contain more complete content
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

# -----------------------
# 4. CREATE ADVANCED RETRIEVER (Hybrid + Rerank)
# -----------------------
@spaces.GPU
def create_advanced_retriever(chunks):
    # We use a significantly more accurate embedding model (bge-small-en-v1.5 or bge-m3 for multilingual text)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", 
        # model_kwargs={'device': 'cpu'}
        model_kwargs={'device':'gpu'}
    )

    # 1. Creating a Vector Database
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    # Increase k to 10 to identify more potential candidates for the Reranker
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2. Creating a keyword search (BM25) for exact matches of words/codes
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    # 3. Combine them into a hybrid search (50% vector weight, 50% text)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    # 4. Add Reranker
    # It will filter out the junk and leave the top 3 most relevant pieces
    compressor = FlashrankRerank(top_n=3)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )

    return compression_retriever

# -----------------------
# 5. CREATE LLM
# -----------------------
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-specdec" # Updated to the latest stable version of Llama 3.3
    )

# -----------------------
# 6. QUERY PIPELINE
# -----------------------
def ask_question(retriever, llm, query):
    # Now we're calling an intelligent compression retriever instead of a simple search
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [f"PAGE {doc.metadata.get('page')}:\n{doc.page_content}" for doc in docs]
    )

    prompt = f"""
You are a strict document assistant.
Answer ONLY using the context below. 
If the answer is not explicitly present in the context, say exactly:
"I could not find the answer in the document."
Do not make up facts, do not use outside knowledge.
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

    print("Loading documents...")
    documents = load_documents(pdf_path)
    
    print("Splitting into chunks...")
    chunks = split_documents(documents)

    print("Building Hybrid Retriever & Reranker (this might take a minute on first run)...")
    retriever = create_advanced_retriever(chunks)
    
    llm = get_llm()
    print("System is ready!")
    
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        if not query.strip():
            continue

        answer = ask_question(retriever, llm, query)
        print("\nAnswer:\n", answer)
