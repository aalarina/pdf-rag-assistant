import gradio as gr
import os
from dotenv import load_dotenv

# Import the logic for your Advanced RAG from another file
from rag_logic import (
    load_documents, 
    split_documents, 
    create_advanced_retriever, 
    get_llm, 
    ask_question
)

load_dotenv()

# A global variable for storing the retriever after a PDF is loaded
retriever_storage = {"instance": None}

# 1. PDF Upload Processing Feature
def process_pdf(file):
    if file is None:
        return "No file selected."
    try:
        # Gradio returns a temporary path to the file
        docs = load_documents(file.name)
        chunks = split_documents(docs)
        
        # Creating and Saving Advanced Retriever
        retriever_storage["instance"] = create_advanced_retriever(chunks)
        return "✅ The PDF has been successfully uploaded and processed! You can now ask questions in the chat below."
    except Exception as e:
        return f"❌ File processing error: {str(e)}"

# 2. A feature for a chatbot
def predict(message, history):
    if retriever_storage["instance"] is None:
        return "Please upload the PDF document in the section above first."
    
    llm = get_llm()
    # Call the search and response function
    response = ask_question(retriever_storage["instance"], llm, message)
    return response

# 3. Building the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Advanced PDF RAG Assistant (Gradio Edition)")
    
    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(label="Load your PDF", file_types=[".pdf"])
        with gr.Column(scale=1):
            upload_button = gr.Button("Process the document", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False, value="Waiting for a file...")
    
    # Link the download button to the processing function
    upload_button.click(
        fn=process_pdf, 
        inputs=file_input, 
        outputs=status_output
    )
    
    gr.Markdown("### 💬 Chat with a document")
    # Gradio's pretty chat interface, ready to use
    gr.ChatInterface(fn=predict)

# Launching the app
if name == "main":
    demo.launch()
