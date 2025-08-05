# ChatBot with Gemini & RAG 🤖

A conversational AI chatbot app built with Streamlit, leveraging Google's Gemini large language model (LLM) and Retrieval-Augmented Generation (RAG) for contextually-aware responses. This app supports both normal conversation mode and document-based Q&A powered by retrieval from your own uploaded text files.

## Features

- **Normal Chat Mode:** Have open-ended conversations with the Gemini LLM, with context-aware responses using memory.
- **RAG-Based Chat Mode:** Upload a `.txt` document and ask questions. The chatbot augments its answers with information retrieved from your document.
- **Session Memory:** Maintains conversation context for more natural interactions.
- **Easy Setup:** Quick Google API key integration and built-in document processing.

## Tech Stack

- **Streamlit:** For the interactive chat web interface.
- **LangChain:** Conversation management, RAG pipeline, and prompt engineering.
- **Gemini (via Google Generative AI):** Powering the large language model.
- **Chroma DB:** Efficient vector storage and retrieval for context-aware responses from user documents.
- **Python:** Main programming language.

## Getting Started

### 1. Clone the Repository

