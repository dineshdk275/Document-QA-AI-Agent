# 📑 Document QA AI Agent

## Overview
The **Document QA AI Agent** is an enterprise-ready, multimodal AI system designed to ingest PDF documents (including resumes, research papers, or reports) and answer questions intelligently. The agent supports:  

- **Text extraction** (structured and unstructured text, including tables).  
- **Image extraction** (figures, charts, and embedded images).  
- **Semantic search** using embeddings for accurate retrieval.  
- **LLM-based question answering** leveraging Gemini API for context-aware responses.  

This tool allows users to quickly extract insights from multiple PDFs without manually scanning documents, making it ideal for HR, research, and enterprise analytics use cases.

---

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

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/document-qa-agent.git
cd document-qa-agent
```
