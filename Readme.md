# 📑 Document QA AI Agent

## Overview
The **Document QA AI Agent** is an enterprise-ready, multimodal AI system designed to ingest PDF documents (including research papers, or reports) and answer questions intelligently. The agent supports:  

- **Text extraction** (structured and unstructured text, including tables).  
- **Image extraction** (figures, charts, and embedded images).  
- **Semantic search** using embeddings for accurate retrieval.  
- **LLM-based question answering** leveraging Gemini API for context-aware responses.  

This tool allows users to quickly extract insights from multiple PDFs without manually scanning documents, making it ideal for HR, research, and enterprise analytics use cases.



## Features

1. **Multi-PDF Ingestion**  
   - Upload multiple PDFs at once.  
   - Extracts text, tables, and images while preserving the document structure.  

2. **Semantic Search & Retrieval**  
   - Text embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`).  
   - Image embeddings using CLIP (`openai/clip-vit-base-patch32`).  
   - Queries return the most relevant text chunks and images.  

3. **Question Answering**  
   - LLM-powered answers using Gemini API (`gemini-1.5-flash`).  
   - Supports extraction of specific metrics or information from documents.  

4. **Enterprise-ready Optimizations**  
   - Context-aware retrieval with ChromaDB.  
   - Handles multiple resumes or research papers in a single workflow.  
   - GPU acceleration for faster image embeddings (optional).  



## Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/document-qa-agent.git
cd document-qa-agent
```
2. **Create and activate a virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Set up environment variables**
**Create a .env file in the project root with:**
```bash
GOOGLE_API_KEY=<your-gemini-api-key>
```

## Usage
1. **Run the Streamlit app**
```bash
streamlit run main.py
```
2. **Upload PDFs**
- Use the sidebar to upload one or multiple PDF files (resumes, research papers, etc.).
- Click Submit & Process to index documents into ChromaDB.
3. **Ask Questions**
- Enter queries in the input box (e.g., “What skills does Candidate A have?” or “Summarize the methodology of Paper B”).
- The agent retrieves relevant text and images, then generates a concise, LLM-based answer.

## Dependencies
- Python 3.10+
- Streamlit
- PyMuPDF (fitz)
- pdfplumber
- Pillow (PIL)
- torch
- sentence-transformers
- transformers
- chromadb
- langchain
- python-dotenv
- langchain-google-genai
## Notes
- Ensure your GOOGLE_API_KEY is valid for Gemini API access.
- For large PDFs or multiple files, GPU acceleration improves performance for image embeddings.
- The system currently supports PDF input only.
  
