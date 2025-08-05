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


### 2. Install Dependencies

It's recommended to use a virtual environment.


### 3. Set up API Key

Create a `secret.py` file in the root directory with your Google API key:


### 3. Set up API Key

Create a `secret.py` file in the root directory with your Google API key:


### 4. Run the App


## Usage

- **Choose Mode:** Use the sidebar to select between "Normal Chat" and "RAG-Based Chat".
- **Upload Document:** In "RAG-Based Chat" mode, upload a `.txt` file. The app will process the file for document-based Q&A.
- **Chat:** Type your questions in the chat input and receive contextually relevant answers.

## Example Screenshot

![Chatbot UI Screenshot](screenshot.png)

## Folder Structure

.
├── app.py
├── requirements.txt
├── secret.py
└── chroma_db/


## Notes

- Only `.txt` documents are currently supported for upload in RAG mode.
- Make sure your Google API key has access to Gemini and Generative AI features.
- Document vectors are persisted in the `chroma_db` directory.

## License

MIT License

---

Feel free to customize this README with more screenshots, project badges, or a features roadmap!
