import streamlit as st
import os
# Import load_documents, split_documents, create_advanced_retriever, get_llm, ask_question

st.set_page_config(page_title="Advanced PDF RAG", layout="centered")
st.title("📚 Complex PDF RAG Assistant")

# Initializing a session to save the retriever's state and chat history
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for uploading a file
with st.sidebar:
    st.header("Loading")
    uploaded_file = st.file_uploader("Choose PDF file", type="pdf")
    
    if uploaded_file and st.session_state.retriever is None:
        with st.spinner("Processing the document... This will take a few moments."):
            # We temporarily save a file for PyPDFLoader
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # We're calling the conveyor
            docs = load_documents("temp.pdf")
            chunks = split_documents(docs)
            st.session_state.retriever = create_advanced_retriever(chunks)
            st.success("The document has been successfully processed!")

# Displaying the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Field for entering a question
if query := st.chat_input("Ask a question about the document..."):
    if st.session_state.retriever is None:
        st.warning("Please upload the PDF document in the sidebar first.")
    else:
        # Displaying the user's question
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate and display the response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = get_llm()
                answer = ask_question(st.session_state.retriever, llm, query)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
