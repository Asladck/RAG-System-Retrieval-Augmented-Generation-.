RAG System (CLI) - Minimal implementation

Usage:
1. Create a Python virtual environment and install requirements:
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt

2. Run the CLI to ingest documents and query:
   python rag_cli.py ingest --source_dir data
   python rag_cli.py query --k 3 "Your question"

This project supports PDF, TXT, MD, DOCX files. Uses sentence-transformers all-MiniLM-L6-v2 for embeddings and FAISS for vector store. If OPENAI_API_KEY is set, uses OpenAI for answer generation; otherwise uses a local transformers T5 model as fallback.
