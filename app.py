# app.py
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import os
import secret

# Set Streamlit page config
st.set_page_config(page_title="ChatBot with Gemini & RAG", page_icon="🤖")

# Initialize session state for messages and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Sidebar for mode selection and API key
with st.sidebar:
    st.title("Settings")
    mode = st.radio("Choose Mode:", ("Normal Chat", "RAG-Based Chat"))
    google_api_key = secret.api_key
    
    # File upload for RAG mode
    if mode == "RAG-Based Chat":
        uploaded_file = st.file_uploader("Upload a document (TXT)", type="txt")
        if uploaded_file:
            # Save the uploaded file temporarily
            with open("temp.txt", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and process the document
            loader = TextLoader("temp.txt")
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            # Initialize embeddings and ChromaDB
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=google_api_key
            )
            st.session_state.vector_store = Chroma.from_documents(
                texts, embeddings, persist_directory="./chroma_db"
            )
            st.success("Document processed successfully!")

# Main chat interface
st.title("ChatBot with Gemini & RAG 🤖")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Initialize Gemini model
    if not google_api_key:
        st.error("Please enter your Google API key!")
        st.stop()
    
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-pro", google_api_key=google_api_key, temperature=0.7
    )

    # Normal Chat Mode
    if mode == "Normal Chat":
        # Initialize conversation chain with memory
        conversation = ConversationChain(
            llm=gemini_llm, memory=st.session_state.memory
        )
        
        # Generate response
        response = conversation.predict(input=prompt)
    
    # RAG-Based Chat Mode
    elif mode == "RAG-Based Chat":
        if not st.session_state.vector_store:
            st.error("Please upload a document first!")
            st.stop()
        
        # Define prompt template
        prompt_template = """
        Use the following context to answer the question. If unsure, say you don't know.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """
        prompt_template = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=gemini_llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template},
        )
        
        # Generate response
        response = qa_chain.run(prompt)
    
    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})