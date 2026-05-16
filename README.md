# PDF RAG Assistant (Retrieval-Augmented Generation)

This project is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about PDF documents using semantic search and large language models.

---

## Features

- Load and process PDF documents
- Split documents into meaningful chunks
- Generate embeddings for semantic search
- Store and retrieve vectors using ChromaDB
- Use LLM (Groq LLaMA 3) for answer generation
- Provide grounded responses based only on retrieved context
- Include page-level citations for traceability

---

## How it works

1. PDF is loaded and split into chunks  
2. Each chunk is converted into embeddings  
3. Embeddings are stored in a vector database (ChromaDB)  
4. User query is embedded and matched with relevant chunks  
5. Retrieved context is passed to an LLM  
6. LLM generates a grounded answer based only on context  

---

## Tech Stack

- Python  
- LangChain  
- ChromaDB  
- Groq API (LLaMA 3)  
- Sentence Transformers  
- PyPDFLoader  

---

## Key Concepts

- Retrieval-Augmented Generation (RAG)  
- Vector embeddings  
- Semantic search  
- Prompt engineering  
- Context grounding  
- Hallucination reduction  

---

## Example

**Question:**  
What are management responsibilities?

**Answer:**  
The system retrieves relevant sections from the PDF and generates a grounded answer with page references.

---

## Note

This project demonstrates an AI system architecture using RAG and is not a model training project.

---

## Author

Built as a learning project in AI Engineering & LLM systems.
