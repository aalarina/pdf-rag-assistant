# PDF RAG Assistant

An advanced Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF document and ask questions about its content.

The system combines **semantic search, keyword-based retrieval, hybrid retrieval and neural reranking** before passing the most relevant context to an LLM. The application is deployed as an interactive **Gradio app on Hugging Face Spaces**.

## Features

*  Upload a PDF document directly through the web interface
*  Recursive document chunking with overlapping chunks
* **Hybrid retrieval** combining:

  * Dense semantic search with BGE embeddings + Chroma
  * Keyword-based search with BM25
* Ensemble retrieval with configurable retriever weights
* **FlashRank neural reranking** to select the most relevant passages
* LLM-based answer generation using **Groq**
* Strict context-grounded answering to reduce hallucinations
* Page numbers included in the retrieved context
* Interactive Gradio interface
* Deployed on Hugging Face Spaces

---

## Architecture

```text
                    ┌─────────────────┐
                    │    PDF Upload   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   PyPDFLoader   │
                    └────────┬────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Recursive Character Splitter│
              │ chunk_size = 800            │
              │ chunk_overlap = 150         │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Chunks      │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
       ┌─────────────────┐       ┌─────────────────┐
       │ BGE Embeddings  │       │   BM25 Search   │
       │                 │       │                 │
       │ BAAI/bge-small  │       │ Keyword-based   │
       └────────┬────────┘       └────────┬────────┘
                │                         │
                ▼                         ▼
       ┌─────────────────┐       ┌─────────────────┐
       │     Chroma      │       │ BM25Retriever   │
       └────────┬────────┘       └────────┬────────┘
                │                         │
                └────────────┬────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Ensemble Retriever  │
                  │      50% / 50%      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ FlashRank Reranker  │
                  │       Top 3          │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Context Construction│
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │      Groq LLM       │
                  │   GPT-OSS-120B      │
                  └──────────┬──────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Final Answer   │
                    └─────────────────┘
```

---

## Retrieval Pipeline

### 1. Document Loading

Uploaded PDFs are processed using `PyPDFLoader`.

Each page is preserved as a separate document object, allowing page information to be included in the final context.

### 2. Chunking

Documents are split using `RecursiveCharacterTextSplitter`:

```python
chunk_size = 800
chunk_overlap = 150
```

The overlap helps preserve context between neighboring chunks.

### 3. Semantic Retrieval

The project uses:

```text
BAAI/bge-small-en-v1.5
```

to generate dense vector embeddings.

The embeddings are stored in **Chroma**, which is then used for semantic similarity search.

The vector retriever returns the top 10 candidates.

### 4. Keyword Retrieval

A separate `BM25Retriever` is used to capture exact keyword matches.

This is particularly useful when the query contains:

* technical terms
* names
* codes
* specific terminology
* exact phrases

It also retrieves the top 10 candidates.

### 5. Hybrid Retrieval

The two retrieval methods are combined using `EnsembleRetriever`:

```python
weights = [0.5, 0.5]
```

This gives equal importance to:

* semantic similarity
* lexical/keyword matching

Using both approaches helps reduce the weaknesses of relying on either retrieval strategy alone.

### 6. Reranking

The retrieved candidates are passed to a FlashRank reranker:

```text
ms-marco-MiniLM-L-12-v2
```

The reranker evaluates the relevance of the retrieved passages to the user's query and keeps the top 3 results.

This produces a smaller and more relevant context for the LLM.

---

## Answer Generation

The final context is passed to a Groq-hosted LLM.

The current model is:

```text
openai/gpt-oss-120b
```

The prompt instructs the model to:

1. Answer only using the retrieved document context.
2. Avoid using outside knowledge.
3. Avoid inventing information.
4. Explicitly state when the answer cannot be found in the document.

If the required information is not present in the retrieved context, the assistant responds:

> "I could not find the answer in the document."

This provides a simple guardrail against unsupported answers.

---

## Tech Stack

| Component         | Technology                     |
| ----------------- | ------------------------------ |
| Language          | Python                         |
| LLM framework     | LangChain                      |
| PDF processing    | PyPDFLoader                    |
| Text splitting    | RecursiveCharacterTextSplitter |
| Embeddings        | BAAI/bge-small-en-v1.5         |
| Vector database   | Chroma                         |
| Keyword retrieval | BM25                           |
| Hybrid retrieval  | EnsembleRetriever              |
| Reranking         | FlashRank                      |
| LLM               | Groq / GPT-OSS-120B            |
| UI                | Gradio                         |
| Deployment        | Hugging Face Spaces            |

---

## Project Structure

```text
advanced-pdf-rag/
│
├── app.py
├── rag_logic.py
├── requirements.txt
├── packages.txt
└── README.md
```

### `app.py`

Contains the Gradio interface and application logic:

* PDF upload
* document processing
* chat interface
* communication with the RAG pipeline

### `rag_logic.py`

Contains the core RAG implementation:

* PDF loading
* document chunking
* embedding generation
* Chroma vector store
* BM25 retrieval
* hybrid retrieval
* FlashRank reranking
* LLM interaction

---

## Environment Variables

The application requires:

```text
GROQ_API_KEY
```

The API key should **never be committed to GitHub**.

For Hugging Face Spaces, add the key through the Space's **Secrets** settings.

---

## Why Hybrid Retrieval?

A purely semantic retriever is good at finding passages with similar meaning, but it can sometimes miss exact terminology.

For example, a query containing:

```text
"ISO 27001"
```

may benefit from an exact keyword match.

On the other hand, a semantic retriever can recognize that:

```text
"How is customer information protected?"
```

and:

```text
"What measures are used to secure client data?"
```

are conceptually similar even when they use different words.

By combining BM25 and vector retrieval, the system can leverage both:

```text
Lexical matching + Semantic similarity
```

before applying a neural reranker.

---

## Design Goals

The project was designed around three main goals:

### Relevance

Use multiple retrieval strategies and reranking to provide the LLM with more relevant context.

### Grounded Answers

Restrict generation to information retrieved from the uploaded document.

### Practical Deployment

Build the system as an interactive application that can be deployed and used through a web interface rather than only as a notebook experiment.

---

## Possible Improvements

Future improvements could include:

* Persistent Chroma storage instead of rebuilding the index after every upload
* Conversation memory for multi-turn document discussions
* Support for multiple documents
* Metadata filtering
* Better multilingual embedding models
* Caching embeddings for repeated documents
* More advanced citation handling
* Document preview alongside retrieved passages

---

## Project Highlights

This project demonstrates practical experience with:

* Retrieval-Augmented Generation
* Vector databases
* Embedding models
* Hybrid information retrieval
* BM25
* Neural reranking
* LangChain
* LLM APIs
* Prompt design
* Hallucination mitigation
* Gradio application development
* Hugging Face Spaces deployment
